from app.config import settings
from app.embeddings import embed_query
from app.vector_store import VectorStore


class Retriever:
    """
    Retrieve relevant document chunks for a user query.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
    ):
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a question.

        Chroma distance is used as the ranking signal.
        Lower distance means a closer match.

        The relevance threshold is optional and is interpreted
        as a maximum acceptable distance.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        k = (
            top_k
            if top_k is not None
            else settings.top_k
        )
        if k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else settings.relevance_threshold
        )

        query_embedding = embed_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            # Lower distance is better.
            if distance > threshold and threshold > 0:
                continue

            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

        return retrieved
