from groq import Groq

from app.config import settings
from app.prompts import SYSTEM_PROMPT


MODEL_PRICING_PER_MILLION = {
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    },
}


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

        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "estimated_cost_usd": None,
        }

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

            usage = getattr(response, "usage", None)

            if usage is not None:
                prompt_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )

                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    0,
                )

                total_tokens = getattr(
                    usage,
                    "total_tokens",
                    prompt_tokens + completion_tokens,
                )

                prompt_details = getattr(
                    usage,
                    "prompt_tokens_details",
                    None,
                )

                cached_tokens = (
                    getattr(
                        prompt_details,
                        "cached_tokens",
                        0,
                    )
                    if prompt_details is not None
                    else 0
                )

                pricing = MODEL_PRICING_PER_MILLION.get(
                    self.model
                )

                estimated_cost_usd = None

                if pricing is not None:
                    uncached_input_tokens = max(
                        prompt_tokens - cached_tokens,
                        0,
                    )

                    estimated_cost_usd = (
                        uncached_input_tokens
                        * pricing["input"]
                        / 1_000_000
                    ) + (
                        cached_tokens
                        * pricing["cached_input"]
                        / 1_000_000
                    ) + (
                        completion_tokens
                        * pricing["output"]
                        / 1_000_000
                    )

                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cached_tokens": cached_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                }

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