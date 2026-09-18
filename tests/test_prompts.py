import pytest

from app.prompts import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)


def test_system_prompt_contains_grounding_rules():
    assert "ONLY using the supplied document context" in SYSTEM_PROMPT
    assert "Do not guess" in SYSTEM_PROMPT
    assert "NOT_FOUND" in SYSTEM_PROMPT


def test_build_grounded_prompt():
    chunks = [
        {
            "text": "The validation F1 score was 0.8772.",
            "metadata": {
                "filename": "research.pdf",
                "page": 7,
            },
        },
        {
            "text": "The IoU score was 0.7810.",
            "metadata": {
                "filename": "research.pdf",
                "page": 7,
            },
        },
    ]

    prompt = build_grounded_prompt(
        "What was the validation F1 score?",
        chunks,
    )

    assert "What was the validation F1 score?" in prompt
    assert "0.8772" in prompt
    assert "0.7810" in prompt
    assert "research.pdf, page 7" in prompt
    assert "NOT_FOUND" in prompt


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        build_grounded_prompt(
            "",
            [
                {
                    "text": "Some context.",
                    "metadata": {},
                }
            ],
        )


def test_empty_context_rejected():
    with pytest.raises(ValueError):
        build_grounded_prompt(
            "What is this?",
            [],
        )
