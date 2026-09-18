from pathlib import Path

import pytest

from app.parser import DocumentParseError, parse_document


def test_parse_txt(tmp_path: Path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "This is the first paragraph.\n\n"
        "This is the second paragraph.",
        encoding="utf-8",
    )

    result = parse_document(str(file_path))

    assert len(result) == 1
    assert result[0]["page"] is None
    assert "first paragraph" in result[0]["text"]
    assert "second paragraph" in result[0]["text"]


def test_parse_unsupported_file(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(DocumentParseError):
        parse_document(str(file_path))


def test_parse_missing_file():
    with pytest.raises(DocumentParseError):
        parse_document("/does/not/exist/sample.txt")


def test_parse_empty_txt(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    result = parse_document(str(file_path))

    assert result == []

def test_parse_pdf(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake pdf")

    class FakePage:
        def extract_text(self):
            return "This is PDF content."

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr(
        "app.parser.PdfReader",
        FakeReader,
    )

    result = parse_document(str(file_path))

    assert len(result) == 1
    assert result[0]["page"] == 1
    assert result[0]["text"] == "This is PDF content."