from pathlib import Path

import pytest

from app.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path: Path):
    return VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )


def test_empty_store(vector_store):
    assert vector_store.count() == 0


def test_add_chunks(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "First chunk.",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "chunk-1",
            "text": "Second chunk.",
            "page": 2,
            "chunk_index": 1,
        },
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    count = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="doc-123",
        filename="sample.pdf",
        file_type="pdf",
    )

    assert count == 2
    assert vector_store.count() == 2


def test_chunk_metadata_is_stored(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "Important information.",
            "page": 7,
            "chunk_index": 0,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="doc-456",
        filename="research.pdf",
        file_type="pdf",
    )

    result = vector_store.get_document_chunks(
        "doc-456"
    )

    assert result["documents"] == [
        "Important information."
    ]

    assert result["metadatas"][0]["document_id"] == "doc-456"
    assert result["metadatas"][0]["filename"] == "research.pdf"
    assert result["metadatas"][0]["file_type"] == "pdf"
    assert result["metadatas"][0]["page"] == 7
    assert result["metadatas"][0]["chunk_index"] == 0


def test_txt_chunk_without_page(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "Text file content.",
            "page": None,
            "chunk_index": 0,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="txt-123",
        filename="notes.txt",
        file_type="txt",
    )

    result = vector_store.get_document_chunks(
        "txt-123"
    )

    metadata = result["metadatas"][0]

    assert metadata["filename"] == "notes.txt"
    assert "page" not in metadata


def test_mismatched_chunks_and_embeddings(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "First chunk.",
            "page": 1,
            "chunk_index": 0,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with pytest.raises(ValueError):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="doc-123",
            filename="sample.pdf",
            file_type="pdf",
        )


def test_delete_document(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "First chunk.",
            "page": 1,
            "chunk_index": 0,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="doc-delete",
        filename="sample.pdf",
        file_type="pdf",
    )

    assert vector_store.count() == 1

    vector_store.delete_document("doc-delete")

    assert vector_store.count() == 0

def test_search(vector_store):
    chunks = [
        {
            "chunk_id": "chunk-0",
            "text": "Machine learning information.",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "chunk-1",
            "text": "Database information.",
            "page": 2,
            "chunk_index": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="search-doc",
        filename="search.txt",
        file_type="txt",
    )

    result = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(result["ids"][0]) == 2
    assert result["ids"][0][0] == "search-doc:chunk-0"
    assert result["documents"][0][0] == (
        "Machine learning information."
    )