SYSTEM_PROMPT = """
You are a document question-answering assistant.

Answer the user's question ONLY using the supplied document context.

Rules:
1. The document context is the only source of truth.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent facts.
5. Do not infer information that is not reasonably supported
   by the supplied context.
6. If the context does not contain enough information to answer
   the question, return exactly: NOT_FOUND
7. If only part of the question can be answered from the context,
   answer only the supported part and clearly state what information
   is not available.
8. Keep the answer concise and factual.
"""


def build_grounded_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build the user prompt containing the question and
    retrieved document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError(
            "At least one retrieved chunk is required."
        )

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.get("metadata", {})

        filename = metadata.get(
            "filename",
            "unknown",
        )

        page = metadata.get("page")

        source = filename

        if page is not None:
            source += f", page {page}"

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
Document context:

{context}

User question:
{question}

Answer using only the document context above.
If the answer is not supported by the context,
return exactly: NOT_FOUND
""".strip()
