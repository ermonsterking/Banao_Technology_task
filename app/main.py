from fastapi import FastAPI

from app.schemas import HealthResponse


app = FastAPI(
    title="RAG Question Answering API",
    description="Document-grounded question answering using RAG.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")