from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_txt_document():
    fake_result = {
        "document_id": "test-document-id",
        "filename": "test.txt",
        "chunks_created": 2,
    }

    with patch(
        "app.main.pipeline.ingest_document",
        return_value=fake_result,
    ):
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.txt",
                    BytesIO(b"This is a test document."),
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == fake_result


def test_upload_pdf_document():
    fake_result = {
        "document_id": "test-pdf-id",
        "filename": "test.pdf",
        "chunks_created": 3,
    }

    with patch(
        "app.main.pipeline.ingest_document",
        return_value=fake_result,
    ):
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(b"fake pdf content"),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == fake_result


def test_upload_unsupported_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "test.docx",
                BytesIO(b"unsupported"),
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 400
    assert "Only PDF and TXT files are supported" in response.json()["detail"]


def test_upload_empty_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.txt",
                BytesIO(b""),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_query_documents():
    fake_result = {
        "answer": "The validation F1 score was 0.8772.",
        "sources": [
            {
                "chunk_id": "document-1:chunk-0",
                "filename": "research.txt",
                "page": 1,
                "text": (
                    "The model achieved a validation "
                    "F1 score of 0.8772."
                ),
            }
        ],
        "latency_ms": 42.5,
    }

    with patch(
        "app.main.pipeline.query",
        return_value=fake_result,
    ):
        response = client.post(
            "/query",
            json={
                "question": "What was the validation F1 score?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "The validation F1 score was 0.8772."
    )
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "research.txt"
    assert data["sources"][0]["page"] == 1
    assert data["latency_ms"] == 42.5

def test_query_documents_not_found_returns_empty_sources():
    fake_result = {
        "answer": "NOT_FOUND",
        "sources": [],
        "latency_ms": 35.2,
    }

    with patch(
        "app.main.pipeline.query",
        return_value=fake_result,
    ):
        response = client.post(
            "/query",
            json={
                "question": "Who was the CEO of the company?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "NOT_FOUND"
    assert data["sources"] == []
    assert data["latency_ms"] == 35.2

def test_query_question_too_short():
    response = client.post(
        "/query",
        json={"question": "Hi"},
    )

    assert response.status_code == 422