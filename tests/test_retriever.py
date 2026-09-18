from unittest.mock import patch

import pytest

from app.retriever import Retriever


class FakeVectorStore:
    def search(self, query_embedding, top_k):
        return {
            "ids": [
                [
                    "doc-1:chunk-0",
                    "doc-1:chunk-1",
                    "doc-1:chunk-2",
                ]
            ],
            "documents": [
                [
                    "Information about machine learning.",
                    "Information about retrieval.",
                    "Information about databases.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "document_id": "doc-1",
                        "filename": "test.txt",
                        "file_type": "txt",
                        "chunk_index": 0,
                    },
                    {
                        "document_id": "doc-1",
                        "filename": "test.txt",
                        "file_type": "txt",
                        "chunk_index": 1,
                    },
                    {
                        "document_id": "doc-1",
                        "filename": "test.txt",
                        "file_type": "txt",
                        "chunk_index": 2,
                    },
                ]
            ],
            "distances": [
                [
                    0.1,
                    0.3,
                    0.8,
                ]
            ],
        }


def test_retrieve_returns_ranked_chunks():
    retriever = Retriever(
        vector_store=FakeVectorStore()
    )

    with patch(
        "app.retriever.embed_query",
        return_value=[0.1, 0.2, 0.3],
    ):
        results = retriever.retrieve(
            "What is machine learning?",
            top_k=3,
        )

    assert len(results) == 3

    assert results[0]["chunk_id"] == "doc-1:chunk-0"
    assert results[0]["distance"] == 0.1

    assert results[1]["chunk_id"] == "doc-1:chunk-1"
    assert results[1]["distance"] == 0.3


def test_retrieve_applies_distance_threshold():
    retriever = Retriever(
        vector_store=FakeVectorStore()
    )

    with patch(
        "app.retriever.embed_query",
        return_value=[0.1, 0.2, 0.3],
    ):
        results = retriever.retrieve(
            "What is machine learning?",
            top_k=3,
            relevance_threshold=0.5,
        )

    assert len(results) == 2

    distances = [
        result["distance"]
        for result in results
    ]

    assert distances == [0.1, 0.3]


def test_zero_threshold_does_not_filter():
    retriever = Retriever(
        vector_store=FakeVectorStore()
    )

    with patch(
        "app.retriever.embed_query",
        return_value=[0.1, 0.2, 0.3],
    ):
        results = retriever.retrieve(
            "What is machine learning?",
            top_k=3,
            relevance_threshold=0.0,
        )

    assert len(results) == 3


def test_empty_question_rejected():
    retriever = Retriever(
        vector_store=FakeVectorStore()
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_invalid_top_k_rejected():
    retriever = Retriever(
        vector_store=FakeVectorStore()
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "What is machine learning?",
            top_k=0,
        )
