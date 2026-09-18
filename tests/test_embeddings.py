from unittest.mock import MagicMock

import pytest

from app import embeddings


def test_embed_documents_empty_input():
    result = embeddings.embed_documents([])

    assert result == []


def test_embed_query_rejects_empty_text():
    with pytest.raises(ValueError):
        embeddings.embed_query("")


def test_embed_documents(monkeypatch):
    fake_model = MagicMock()

    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: fake_model,
    )

    result = embeddings.embed_documents(
        [
            "First document chunk.",
            "Second document chunk.",
        ]
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    fake_model.encode.assert_called_once()


def test_embed_query(monkeypatch):
    fake_model = MagicMock()

    fake_model.encode.return_value.tolist.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: fake_model,
    )

    result = embeddings.embed_query(
        "What is the reported F1 score?"
    )

    assert result == [0.1, 0.2, 0.3]

    fake_model.encode.assert_called_once()
