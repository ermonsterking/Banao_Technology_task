import time
import uuid
from pathlib import Path

from app.config import settings
from app.chunker import chunk_document
from app.embeddings import embed_documents
from app.llm import GroundedLLM
from app.parser import parse_document
from app.prompts import build_grounded_prompt
from app.retriever import Retriever
from app.vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        llm: GroundedLLM | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(self.vector_store)
        self.llm = llm or GroundedLLM()

    def ingest_document(
        self,
        file_path: str | Path,
        filename: str,
    ) -> dict:
        path = Path(file_path)

        parsed_pages = parse_document(path)

        if not parsed_pages:
            raise ValueError("Document contains no readable text.")

        chunks = chunk_document(
            parsed_pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            )

        if not chunks:
            raise ValueError("Document produced no chunks.")

        embeddings = embed_documents(
            [chunk["text"] for chunk in chunks]
        )

        document_id = str(uuid.uuid4())

        file_type = path.suffix.lower().lstrip(".")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
        }

    def query(self, question: str) -> dict:
        start_time = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(question)

        if not retrieved_chunks:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "answer": (
                    "I could not find the answer in "
                    "the uploaded documents."
                ),
                "sources": [],
                "latency_ms": round(latency_ms, 2),
            }

        prompt = build_grounded_prompt(
            question,
            retrieved_chunks,
        )

        answer = self.llm.generate(
            question=question,
            context_prompt=prompt,
        )

        sources = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})

            sources.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "filename": metadata.get(
                        "filename",
                        "unknown",
                    ),
                    "page": metadata.get("page"),
                    "text": chunk["text"],
                }
            )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
        }
