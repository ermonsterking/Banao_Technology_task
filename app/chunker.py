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
    2. Detect section headings and keep them with following content.
    3. Split remaining content into paragraphs.
    4. Group semantic units until the target size is reached.
    5. Split oversized units using sentences, then words.
    6. Apply controlled overlap without exceeding chunk_size.
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

        semantic_units = _split_into_semantic_units(page_text)

        for unit in semantic_units:
            if len(unit) <= chunk_size:
                parts = [unit]
            else:
                parts = _split_large_text(
                    unit,
                    chunk_size,
                )

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                if (
                    not chunks
                    or chunks[-1]["page"] != page_number
                ):
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

                separator = "\n\n"

                candidate = (
                    previous_text
                    + separator
                    + part
                )

                if len(candidate) <= chunk_size:
                    chunks[-1]["text"] = candidate
                    continue

                overlap_text = _get_overlap_text(
                    previous_text,
                    chunk_overlap,
                )

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


def _split_into_semantic_units(text: str) -> list[str]:
    """
    Split text into semantic units while keeping a likely section
    heading attached to its following paragraph.

    A heading is identified conservatively using:
    - a short single line
    - no sentence-ending punctuation
    - title-like capitalization or common section-heading wording
    """

    paragraphs = _split_paragraphs(text)

    units = []
    index = 0

    while index < len(paragraphs):
        current = paragraphs[index]

        if (
            _is_section_heading(current)
            and index + 1 < len(paragraphs)
        ):
            units.append(
                f"{current}\n\n{paragraphs[index + 1]}"
            )
            index += 2
        else:
            units.append(current)
            index += 1

    return units


def _is_section_heading(text: str) -> bool:
    """Return True when text looks like a section heading."""

    text = text.strip()

    if not text:
        return False

    # Headings should generally be short.
    if len(text) > 100:
        return False

    # A normal sentence is unlikely to be a heading.
    if re.search(r"[.!?]$", text):
        return False

    words = text.split()

    if len(words) > 12:
        return False

    # Common heading signals.
    heading_keywords = {
        "overview",
        "architecture",
        "training",
        "preprocessing",
        "evaluation",
        "results",
        "tools",
        "deployment",
        "limitation",
        "methodology",
        "introduction",
        "conclusion",
        "implementation",
        "dataset",
        "experiments",
    }

    lowered_words = {word.lower().strip(":") for word in words}

    if lowered_words & heading_keywords:
        return True

    # Title-like heading: most words begin with uppercase.
    title_like_words = [
        word for word in words
        if word[:1].isupper()
    ]

    return (
        len(words) <= 8
        and len(title_like_words) >= max(2, len(words) // 2)
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split text while preserving paragraph boundaries."""

    paragraphs = re.split(
        r"\n\s*\n+",
        text,
    )

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
    sentences → words.
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

    return _split_by_words(
        text,
        chunk_size,
    )


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