from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.llm import GroundedLLM, LLMError


def test_missing_api_key():
    with pytest.raises(LLMError):
        GroundedLLM(
            api_key="",
            model="test-model",
        )


def test_missing_model():
    with pytest.raises(LLMError):
        GroundedLLM(
            api_key="test-key",
            model="",
        )


def test_generate(monkeypatch):
    fake_client = MagicMock()

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="The validation F1 score was 0.8772."
                )
            )
        ]
    )

    fake_client.chat.completions.create.return_value = (
        fake_response
    )

    llm = GroundedLLM(
        api_key="test-key",
        model="test-model",
    )

    monkeypatch.setattr(
        llm,
        "client",
        fake_client,
    )

    answer = llm.generate(
        question="What was the validation F1 score?",
        context_prompt=(
            "The validation F1 score was 0.8772."
        ),
    )

    assert answer == (
        "The validation F1 score was 0.8772."
    )

    fake_client.chat.completions.create.assert_called_once()


def test_empty_question_rejected():
    llm = GroundedLLM(
        api_key="test-key",
        model="test-model",
    )

    with pytest.raises(ValueError):
        llm.generate(
            question="",
            context_prompt="Some context.",
        )


def test_empty_context_rejected():
    llm = GroundedLLM(
        api_key="test-key",
        model="test-model",
    )

    with pytest.raises(ValueError):
        llm.generate(
            question="What is this?",
            context_prompt="",
        )


def test_api_failure_wrapped():
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = (
        RuntimeError("API unavailable")
    )

    llm = GroundedLLM(
        api_key="test-key",
        model="test-model",
    )

    llm.client = fake_client

    with pytest.raises(LLMError, match="LLM generation failed"):
        llm.generate(
            question="What is this?",
            context_prompt="Some context.",
        )
