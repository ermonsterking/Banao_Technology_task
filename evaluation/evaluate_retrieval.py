import json
import tempfile
from pathlib import Path

from app.chunker import chunk_document
from app.config import settings
from app.embeddings import embed_documents
from app.parser import parse_document
from app.retriever import Retriever
from app.vector_store import VectorStore


EVALUATION_DIR = Path(__file__).parent
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
EVALUATION_DOCUMENT = EVALUATION_DIR / "retrieval_eval.txt"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_evaluation_retriever(
    persist_directory: str,
) -> Retriever:
    """Build a fresh vector store from the committed evaluation document."""

    if not EVALUATION_DOCUMENT.exists():
        raise FileNotFoundError(
            f"Evaluation document not found: {EVALUATION_DOCUMENT}"
        )

    pages = parse_document(EVALUATION_DOCUMENT)

    chunks = chunk_document(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_documents(texts)

    vector_store = VectorStore(
        persist_directory=persist_directory,
    )

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="retrieval-evaluation",
        filename=EVALUATION_DOCUMENT.name,
        file_type=".txt",
    )

    return Retriever(vector_store)


def evaluate_retrieval(
    questions: list[dict],
    top_k: int,
) -> dict:
    with tempfile.TemporaryDirectory() as temporary_directory:
        retriever = build_evaluation_retriever(
            temporary_directory
        )

        results = []

        for item in questions:
            question = item["question"]
            answerable = item["answerable"]
            expected_text = item["expected_text"]

            retrieved = retriever.retrieve(
                question,
                top_k=top_k,
                relevance_threshold=0.0,
            )

            retrieved_text = " ".join(
                chunk["text"].lower()
                for chunk in retrieved
            )

            hit = (
                answerable
                and expected_text.lower() in retrieved_text
            )

            distances = [
                round(chunk["distance"], 4)
                for chunk in retrieved
            ]

            results.append(
                {
                    "question": question,
                    "answerable": answerable,
                    "hit": hit,
                    "distances": distances,
                }
            )

        answerable_questions = [
            result for result in results
            if result["answerable"]
        ]

        retrieval_hits = sum(
            result["hit"]
            for result in answerable_questions
        )

        total_answerable = len(answerable_questions)

        return {
            "top_k": top_k,
            "hits": retrieval_hits,
            "total_answerable": total_answerable,
            "hit_rate": (
                retrieval_hits / total_answerable
                if total_answerable
                else 0.0
            ),
            "results": results,
        }


if __name__ == "__main__":
    questions = load_questions()

    print("\nRetrieval Evaluation")
    print("=" * 70)

    for top_k in [1, 3, 4]:
        evaluation = evaluate_retrieval(
            questions,
            top_k=top_k,
        )

        print(
            f"\nHit Rate @ {top_k}: "
            f"{evaluation['hit_rate']:.2%} "
            f"({evaluation['hits']}/"
            f"{evaluation['total_answerable']})"
        )

        for result in evaluation["results"]:
            status = (
                "PASS"
                if result["hit"]
                else "N/A"
                if not result["answerable"]
                else "FAIL"
            )

            distances = ", ".join(
                str(distance)
                for distance in result["distances"]
            )

            label = (
                "answerable"
                if result["answerable"]
                else "unanswerable"
            )

            print(
                f"{status} | {label} | "
                f"{result['question']}"
            )
            print(
                f"      distances: [{distances}]"
            )