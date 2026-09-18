import pytest

from app.chunker import chunk_document


def test_short_document_creates_one_chunk():
    pages = [
        {
            "text": (
                "This is the first paragraph.\n\n"
                "This is the second paragraph."
            ),
            "page": 1,
        }
    ]

    chunks = chunk_document(
        pages,
        chunk_size=700,
        chunk_overlap=120,
    )

    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert "first paragraph" in chunks[0]["text"]
    assert "second paragraph" in chunks[0]["text"]


def test_large_document_creates_multiple_chunks():
    text = (
        "This is a paragraph containing useful information. "
        * 100
    )

    pages = [
        {
            "text": text,
            "page": 1,
        }
    ]

    chunks = chunk_document(
        pages,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk["text"]) <= 300


def test_page_metadata_is_preserved():
    pages = [
        {
            "text": "Information from page one.",
            "page": 1,
        },
        {
            "text": "Information from page two.",
            "page": 2,
        },
    ]

    chunks = chunk_document(
        pages,
        chunk_size=700,
        chunk_overlap=120,
    )

    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2


def test_chunk_ids_are_unique():
    pages = [
        {
            "text": (
                "Information. " * 100
            ),
            "page": 1,
        }
    ]

    chunks = chunk_document(
        pages,
        chunk_size=100,
        chunk_overlap=20,
    )

    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_document(
            [{"text": "hello", "page": 1}],
            chunk_size=0,
            chunk_overlap=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document(
            [{"text": "hello", "page": 1}],
            chunk_size=100,
            chunk_overlap=100,
        )
