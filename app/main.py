import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.llm import LLMError
from app.rag_pipeline import RAGPipeline
from app.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

app = FastAPI(
    title="RAG Question Answering API",
    description="Document-grounded question answering using RAG.",
    version="0.2.0",
)

pipeline = RAGPipeline()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            result = pipeline.ingest_document(
                file_path=temp_path,
                filename=file.filename,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        return UploadResponse(**result)

    except (ValueError, OSError) as exc:
        logger.warning(
            "Document ingestion rejected: filename=%s error=%s",
            file.filename,
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "Unexpected document ingestion failure: filename=%s",
            file.filename,
        )
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_documents(
    request: QueryRequest,
) -> QueryResponse:

    try:
        result = pipeline.query(request.question)

        return QueryResponse(**result)

    except LLMError as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="LLM generation failed.",
        ) from exc

    except ValueError as exc:
        logger.warning(
            "Invalid query request: error=%s",
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected query failure.")
        raise HTTPException(
            status_code=500,
            detail="Query failed.",
        )