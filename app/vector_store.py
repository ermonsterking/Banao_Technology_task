from pathlib import Path
from typing import Optional

import chromadb

from app.config import settings


COLLECTION_NAME = "rag_documents"


class VectorStore:
    """Persistent ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
    ):
        directory = persist_directory or settings.chroma_dir

        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=directory
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Document chunks for RAG question answering"
            },
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
        file_type: str,
    ) -> int:
        """
        Store document chunks, embeddings, and metadata.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            # Prefix the document ID to prevent collisions
            # between different uploaded documents.
            vector_id = f"{document_id}:{chunk_id}"

            ids.append(vector_id)
            documents.append(chunk["text"])

            metadata = {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": chunk["chunk_index"],
            }

            # Chroma metadata does not handle None reliably,
            # so only include page when it exists.
            if chunk.get("page") is not None:
                metadata["page"] = chunk["page"]

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    def get_document_chunks(
        self,
        document_id: str,
    ) -> dict:
        """Return all chunks belonging to a document."""

        return self.collection.get(
            where={
                "document_id": document_id
            }
        )

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete all chunks belonging to a document."""

        self.collection.delete(
            where={
                "document_id": document_id
            }
        )
