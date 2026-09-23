import logging
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

logger = logging.getLogger(__name__)

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
        logger.info(
            "Document parsed and chunked: filename=%s chunks=%d",
            filename,
            len(chunks),
        )

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
        logger.info(
            "Document ingested: filename=%s document_id=%s chunks=%d",
            filename,
            document_id,
            len(chunks),
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
        }


    def query(self, question: str) -> dict:
        start_time = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(question)
        logger.info(
            "Query retrieval completed: retrieved_chunks=%d",
            len(retrieved_chunks),
        )

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
        usage = self.llm.last_usage

        # Do not expose retrieved chunks as sources when
        # the LLM determines that the answer is not supported.
        if answer.strip() == "NOT_FOUND":
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.info(
                (
                    "Query completed without sufficient evidence: "
                    "retrieved_chunks=%d latency_ms=%.2f "
                    "prompt_tokens=%d completion_tokens=%d "
                    "total_tokens=%d cached_tokens=%d "
                    "estimated_cost_usd=%s"
                ),
                len(retrieved_chunks),
                latency_ms,
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["total_tokens"],
                usage["cached_tokens"],
                (
                    f"{usage['estimated_cost_usd']:.8f}"
                    if usage["estimated_cost_usd"] is not None
                    else "unavailable"
                ),
            )

            return {
                "answer": "NOT_FOUND",
                "sources": [],
                "latency_ms": round(latency_ms, 2),
            }

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

        logger.info(
            (
                "Query completed: retrieved_chunks=%d "
                "latency_ms=%.2f prompt_tokens=%d "
                "completion_tokens=%d total_tokens=%d "
                "cached_tokens=%d estimated_cost_usd=%s"
            ),
            len(retrieved_chunks),
            latency_ms,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            usage["cached_tokens"],
            (
                f"{usage['estimated_cost_usd']:.8f}"
                if usage["estimated_cost_usd"] is not None
                else "unavailable"
            ),
        )

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
        }
    
