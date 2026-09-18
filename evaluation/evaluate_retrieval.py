import json
from pathlib import Path

from app.retriever import Retriever


QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(
    questions: list[dict],
    top_k: int = 3,
) -> dict:

    retriever = Retriever()

    answerable_questions = [
        item for item in questions
        if item["answerable"]
    ]

    unanswerable_questions = [
        item for item in questions
        if not item["answerable"]
    ]

    retrieval_hits = 0
    results = []

    for item in questions:
        question = item["question"]
        expected_text = item.get("expected_text")

        retrieved = retriever.retrieve(
            question,
            top_k=top_k,
            relevance_threshold=0.0,
        )

        retrieved_text = " ".join(
            chunk["text"].lower()
            for chunk in retrieved
        )

        if expected_text is not None:
            hit = expected_text.lower() in retrieved_text
        else:
            hit = False

        if item["answerable"] and hit:
            retrieval_hits += 1

        results.append(
            {
                "question": question,
                "answerable": item["answerable"],
                "expected_text": expected_text,
                "hit": hit,
                "retrieved_count": len(retrieved),
                "distances": [
                    round(chunk["distance"], 4)
                    for chunk in retrieved
                ],
            }
        )

    total_answerable = len(answerable_questions)

    retrieval_hit_rate = (
        retrieval_hits / total_answerable
        if total_answerable > 0
        else 0.0
    )

    return {
        "top_k": top_k,
        "retrieval_hits": retrieval_hits,
        "answerable_total": total_answerable,
        "retrieval_hit_rate": retrieval_hit_rate,
        "unanswerable_total": len(unanswerable_questions),
        "results": results,
    }


if __name__ == "__main__":
    questions = load_questions()

    evaluation = evaluate_retrieval(
        questions,
        top_k=3,
    )

    print("\nRetrieval Evaluation")
    print("=" * 60)

    print(
        f"Retrieval Hit Rate @ {evaluation['top_k']}: "
        f"{evaluation['retrieval_hit_rate']:.2%}"
    )

    print(
        f"Answerable retrieval hits: "
        f"{evaluation['retrieval_hits']}/"
        f"{evaluation['answerable_total']}"
    )

    print(
        f"Unanswerable questions: "
        f"{evaluation['unanswerable_total']}"
    )

    print("\nIndividual Results")
    print("-" * 60)

    for result in evaluation["results"]:
        print(
            f"\nQuestion: {result['question']}"
        )
        print(
            f"Answerable: {result['answerable']}"
        )
        print(
            f"Retrieval Hit: {result['hit']}"
        )
        print(
            f"Distances: {result['distances']}"
        )