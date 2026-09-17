from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask about the uploaded documents.",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    filename: str
    page: Optional[int] = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int


class HealthResponse(BaseModel):
    status: str