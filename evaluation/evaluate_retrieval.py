import json
from pathlib import Path

from app.retriever import Retriever


QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(
    questions: list[dict],
    top_k: int,
) -> dict:
    retriever = Retriever()

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

