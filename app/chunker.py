import re


def chunk_document(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[dict]:
    """
    Convert page-level document text into semantically meaningful chunks.

    Strategy:
    1. Preserve page boundaries.
    2. Split text into paragraphs.
    3. Group paragraphs until the target size is reached.
    4. Split oversized paragraphs into sentences.
    5. Fall back to word-level splitting when necessary.
    6. Apply overlap without exceeding chunk_size.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []
    chunk_index = 0

    for page in pages:
        page_text = page.get("text", "").strip()
        page_number = page.get("page")

        if not page_text:
            continue

        paragraphs = _split_paragraphs(page_text)

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                parts = [paragraph]
            else:
                parts = _split_large_text(
                    paragraph,
                    chunk_size,
                )

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                # If this is the first part on the page,
                # simply create a chunk.
                if not chunks or chunks[-1]["page"] != page_number:
                    chunks.append(
                        {
                            "chunk_id": f"chunk-{chunk_index}",
                            "text": part,
                            "page": page_number,
                            "chunk_index": chunk_index,
                        }
                    )

                    chunk_index += 1
                    continue

                previous_text = chunks[-1]["text"]

                # Try extending the previous chunk.
                separator = "\n\n"
                candidate = (
                    previous_text
                    + separator
                    + part
                )

                if len(candidate) <= chunk_size:
                    chunks[-1]["text"] = candidate
                    continue

                # Previous chunk is full enough.
                # Create a new chunk with controlled overlap.
                overlap_text = _get_overlap_text(
                    previous_text,
                    chunk_overlap,
                )

                if overlap_text:
                    available = (
                        chunk_size
                        - len(overlap_text)
                        - len(separator)
                    )

                    if available > 0:
                        part = part[:available].rstrip()

                chunks.append(
                    {
                        "chunk_id": f"chunk-{chunk_index}",
                        "text": (
                            f"{overlap_text}{separator}{part}"
                            if overlap_text
                            else part
                        ),
                        "page": page_number,
                        "chunk_index": chunk_index,
                    }
                )

                chunk_index += 1

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text while preserving paragraph boundaries."""

    paragraphs = re.split(r"\n\s*\n+", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def _split_large_text(
    text: str,
    chunk_size: int,
) -> list[str]:
    """
    Split oversized text using:
    sentences → words → character slices.
    """

    sentences = _split_sentences(text)

    if len(sentences) > 1:
        parts = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > chunk_size:
                if current:
                    parts.append(current)
                    current = ""

                parts.extend(
                    _split_by_words(
                        sentence,
                        chunk_size,
                    )
                )
                continue

            if not current:
                current = sentence
                continue

            candidate = f"{current} {sentence}"

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                parts.append(current)
                current = sentence

        if current:
            parts.append(current)

        return parts

    return _split_by_words(text, chunk_size)


def _split_sentences(text: str) -> list[str]:
    """Basic sentence splitting."""

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


def _split_by_words(
    text: str,
    chunk_size: int,
) -> list[str]:
    """Split text at word boundaries."""

    words = text.split()

    parts = []
    current_words = []
    current_length = 0

    for word in words:
        additional_length = (
            len(word)
            if not current_words
            else len(word) + 1
        )

        if (
            current_words
            and current_length + additional_length > chunk_size
        ):
            parts.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def _get_overlap_text(
    text: str,
    overlap: int,
) -> str:
    """
    Return approximately `overlap` characters from the end
    of the previous chunk, preferring a word boundary.
    """

    if overlap <= 0:
        return ""

    if len(text) <= overlap:
        return text

    candidate = text[-overlap:]

    first_space = candidate.find(" ")

    if first_space != -1:
        candidate = candidate[first_space + 1:]

    return candidate.strip()
