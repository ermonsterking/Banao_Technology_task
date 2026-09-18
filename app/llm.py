from groq import Groq

from app.config import settings
from app.prompts import SYSTEM_PROMPT


class LLMError(Exception):
    """Raised when the LLM cannot generate an answer."""


class GroundedLLM:
    """Groq-backed LLM for document-grounded generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = (
            api_key
            if api_key is not None
            else settings.groq_api_key
        )

        self.model = (
            model
            if model is not None
            else settings.groq_model
        )

        if not self.api_key:
            raise LLMError(
                "GROQ_API_KEY is not configured."
            )

        if not self.model:
            raise LLMError(
                "GROQ_MODEL is not configured."
            )

        try:
            self.client = Groq(
                api_key=self.api_key
            )
        except Exception as exc:
            raise LLMError(
                f"Failed to initialize Groq client: {exc}"
            ) from exc

    def generate(
        self,
        question: str,
        context_prompt: str,
    ) -> str:
        """
        Generate a grounded answer using the supplied context.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not context_prompt or not context_prompt.strip():
            raise ValueError(
                "Context prompt cannot be empty."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": context_prompt,
                    },
                ],
                temperature=0,
            )

            answer = response.choices[0].message.content

            if not answer or not answer.strip():
                raise LLMError(
                    "LLM returned an empty response."
                )

            return answer.strip()

        except LLMError:
            raise

        except Exception as exc:
            raise LLMError(
                f"LLM generation failed: {exc}"
            ) from exc
