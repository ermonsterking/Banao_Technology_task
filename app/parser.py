from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed."""


def parse_document(file_path: str) -> list[dict]:
    """
    Parse a PDF or TXT document.

    Returns a list of dictionaries containing extracted text
    and page metadata.
    """
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(
            f"Unsupported file type: {path.suffix}"
        )

    if not path.exists():
        raise DocumentParseError(
            f"File does not exist: {file_path}"
        )

    try:
        if path.suffix.lower() == ".pdf":
            return _parse_pdf(path)

        return _parse_txt(path)

    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Failed to parse {path.name}: {exc}"
        ) from exc


def _parse_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages


def _parse_txt(path: Path) -> list[dict]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    text = _clean_text(text)

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None,
        }
    ]


def _clean_text(text: str) -> str:
    """
    Perform light text normalization while preserving
    paragraph boundaries.
    """
    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")

            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()
