# RAG Question Answering

A document-grounded Retrieval-Augmented Generation (RAG) system that allows users to upload PDF or TXT documents and ask questions about their contents.

The system parses documents, creates structure-aware chunks, generates semantic embeddings, stores them in a persistent local vector database, retrieves relevant chunks for a question, and generates an answer using a Groq-hosted LLM.

The generation step is explicitly grounded in the retrieved document context. When the available context does not contain enough information to answer a question, the system returns `NOT_FOUND` instead of asking the LLM to rely on outside knowledge.

---

## Overview

This project implements an end-to-end RAG pipeline using:

* Python
* FastAPI
* Pydantic
* ChromaDB
* Sentence Transformers
* BGE-small embeddings
* Groq LLM
* PyPDF

### Core Workflow

```text
PDF / TXT Document
        ↓
Document Parsing
        ↓
Text Cleaning
        ↓
Structure-Aware Chunking
        ↓
BGE-small Embeddings
        ↓
Persistent ChromaDB
        ↓
Query Embedding
        ↓
Top-K Retrieval
        ↓
Grounded LLM Generation
        ↓
Answer + Source Chunks
```

---

# Features

* PDF and TXT document upload
* PDF page-level metadata preservation
* Text cleaning and normalization
* Structure-aware document chunking
* Semantic embedding generation
* Persistent local ChromaDB vector store
* Top-K semantic retrieval
* Groq-based grounded answer generation
* Source chunk attribution
* Out-of-document question handling
* `NOT_FOUND` abstention behavior
* Pydantic request/response validation
* Error handling for ingestion and LLM failures
* Operational logging
* Automated test suite
* Retrieval-quality evaluation

---

# Architecture & Design Decisions

## 1. Document Parsing

The system currently supports:

* PDF files using `PyPDF`
* TXT files using Python's built-in text handling

Extracted text is lightly normalized while preserving paragraph boundaries.

For PDFs, page-level metadata is retained so that retrieved chunks can be traced back to their original page.

The current PDF parser is text-extraction based and does not perform OCR.

---

## 2. Structure-Aware Chunking

Chunking is one of the main design considerations in this RAG system.

Instead of blindly splitting text at fixed character boundaries, the implementation attempts to preserve meaningful document structure.

The chunking pipeline:

1. Preserves page boundaries.
2. Detects section headings.
3. Keeps headings together with their following content.
4. Splits remaining content into paragraphs.
5. Groups semantic units until the target size is reached.
6. Splits oversized units using sentences and then words.
7. Applies controlled overlap between adjacent chunks.

### Current Configuration

| Parameter     |          Value |
| ------------- | -------------: |
| Chunk size    | 700 characters |
| Chunk overlap | 120 characters |

The values were selected to balance semantic context with retrieval precision while keeping the amount of context passed to the LLM reasonably compact.

The structure-aware strategy was also validated through retrieval evaluation. Details are provided in the evaluation section below.

---

## 3. Embeddings

The system uses:

```text
BAAI/bge-small-en-v1.5
```

The embedding model converts both document chunks and user questions into dense vector representations.

Embeddings are normalized before storage and retrieval.

The embedding model is loaded once per application process using caching, avoiding repeated model initialization for every request.

---

## 4. Vector Store

The project uses persistent local **ChromaDB**.

Each document chunk is stored with metadata including:

* Document ID
* Filename
* File type
* Chunk index
* Page number when available

This metadata allows retrieved chunks to be returned together with their source information.

The local vector database is stored under:

```text
chroma_db/
```

The database contents are excluded from version control.

---

## 5. Retrieval

For every user question:

1. The question is converted into an embedding.
2. ChromaDB performs semantic similarity search.
3. The top relevant chunks are retrieved.
4. The retrieved chunks are provided to the grounded LLM.

### Current Retrieval Configuration

| Parameter            |    Value |
| -------------------- | -------: |
| Top-K                |        3 |
| Similarity threshold | Disabled |

ChromaDB returns distances where lower values represent closer matches.

### Why Top-K = 3?

Retrieval quality was evaluated using six answerable questions.

| Top-K | Answerable Questions | Hits | Hit Rate |
| ----: | -------------------: | ---: | -------: |
|     1 |                    6 |    4 |      66.67% |
|     3 |                    6 |    6 |     100% |
|     4 |                    6 |    6 |     100% |

Increasing Top-K from 1 to 3 substantially improved retrieval coverage.

Increasing it from 3 to 4 did not improve the measured result.

Therefore, `Top-K = 3` was selected to provide the observed retrieval coverage while keeping the retrieved context relatively small.

### Why Is the Similarity Threshold Disabled?

A simple fixed distance threshold was experimentally evaluated.

The observed distance distributions for answerable and unanswerable questions overlapped. Some answerable questions produced relatively high distances, while some unanswerable questions produced relatively low distances.

Because of this overlap, a fixed threshold could incorrectly reject relevant chunks or retain irrelevant chunks.

The current implementation therefore retrieves the top 3 candidates and relies on the grounded LLM to determine whether the retrieved context actually supports the answer.

---

## 6. Grounded Generation

Retrieved chunks are passed to a Groq-hosted LLM together with a strict grounding instruction.

The model is instructed to:

* Use only the supplied document context.
* Avoid outside knowledge.
* Avoid guessing.
* Avoid inventing facts.
* Return `NOT_FOUND` when the context does not contain enough information.

The LLM temperature is set to `0` to make responses more deterministic.

The document context is explicitly treated as the source of truth for answer generation.

---

## 7. Source Attribution

For supported questions, the API returns source chunks containing:

* Chunk ID
* Filename
* Page number when available
* Chunk text

When the LLM returns `NOT_FOUND`, the API returns an empty source list.

This prevents retrieved but unsupported chunks from being presented as evidence for an answer that the context does not actually support.

---

# API Documentation

The application exposes a FastAPI REST API.

## Health Check

### Endpoint

```text
GET /health
```

Checks whether the API is running.

### Example

```bash
curl http://127.0.0.1:8000/health
```

### Response

```json
{
  "status": "ok"
}
```

---

## Upload Document

### Endpoint

```text
POST /documents/upload
```

Uploads a PDF or TXT document and adds its chunks to the vector store.

### Supported Files

* `.pdf`
* `.txt`

### Example

```bash
curl -X POST \
  -F "file=@data/example.pdf" \
  http://127.0.0.1:8000/documents/upload
```

### Example Response

```json
{
  "document_id": "generated-document-id",
  "filename": "example.pdf",
  "chunks_created": 12
}
```

The exact document ID and number of chunks depend on the uploaded document.

---

## Ask a Question

### Endpoint

```text
POST /query
```

### Request

```json
{
  "question": "What is the validation F1 score?"
}
```

### Example

```bash
curl -X POST \
  http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the validation F1 score?"}'
```

### Example Response

```json
{
  "answer": "The validation F1 score is 0.8772.",
  "sources": [
    {
      "chunk_id": "chunk_0",
      "filename": "example.pdf",
      "page": 4,
      "text": "The model achieved a validation F1 score of 0.8772..."
    }
  ],
  "latency_ms": 700.25
}
```

The exact values will vary depending on the uploaded document and execution environment.

---

## Out-of-Document Questions

If the uploaded documents do not contain enough information to answer a question, the system returns:

```json
{
  "answer": "NOT_FOUND",
  "sources": [],
  "latency_ms": 650.12
}
```

This is intentional.

For example, if the document contains information about a machine-learning project but does not mention the project's CEO, asking:

```text
Who was the CEO of the company?
```

should not cause the system to generate an answer from the model's general knowledge.

---

# Request Validation

The API uses Pydantic models for request and response validation.

The `/query` endpoint validates that:

* A question is provided.
* The question contains at least 3 characters.
* The question contains no more than 1000 characters.

Invalid requests are rejected before reaching the RAG pipeline.

---

# Error Handling

Errors are handled at different stages of the application.

## Document Ingestion

The API validates:

* Filename presence
* Supported file extension
* Empty uploads
* Readable document content
* Successful chunk generation

Invalid document requests return HTTP `400`.

Unexpected ingestion failures return HTTP `500` with a safe error message.

## LLM Generation

LLM initialization and generation failures are represented using a dedicated `LLMError` exception.

The API handles these failures without intentionally exposing internal implementation details or credentials.

---

# Retrieval Evaluation

A small manually curated evaluation dataset is included under:

```text
evaluation/
├── questions.json
└── evaluate_retrieval.py
```

The evaluation currently contains:

* 6 answerable questions
* 5 unanswerable questions

For answerable questions, a retrieval hit is counted when the expected answer-bearing text appears within the retrieved chunks.

### Final Top-K Results

| Top-K | Answerable Questions | Hits | Hit Rate |
| ----: | -------------------: | ---: | -------: |
|     1 |                    6 |    4 |     66.67% |
|     3 |                    6 |    6 |     100% |
|     4 |                    6 |    6 |     100% |

Run the evaluation with:

```bash
python -m evaluation.evaluate_retrieval
```

The evaluation set is intentionally small and should not be interpreted as a general benchmark of RAG retrieval performance.

---

# Observed Failure and Improvement

An important retrieval failure was observed during development.

The architecture-related question initially failed to retrieve the relevant information within the top 3 results. The relevant information appeared at rank 4.

## Root Cause

The document contained a section heading describing the model architecture, but the initial chunking strategy did not sufficiently preserve the relationship between the heading and the content that followed it.

This reduced the semantic relevance of the resulting chunks.

## Change Made

The chunking strategy was changed to preserve document structure.

The updated chunker:

* Detects section headings.
* Keeps headings with their following content.
* Preserves page boundaries.
* Groups related paragraphs together.
* Splits oversized semantic units only when necessary.

## Result

In the six-question evaluation used during this experiment, Hit Rate@3 improved from:

```text
83.33% (5/6)
```

to:

```text
100% (6/6)
```

This experiment was the primary reason for retaining the structure-aware chunking approach.

---

# Performance Observations

The application records request latency in milliseconds.

During local testing, the first query was significantly slower because the embedding model needed to be initialized.

Subsequent queries were faster because the embedding model was cached within the application process.

A small local test produced approximately:

* First query: 8.4 seconds
* Subsequent warm queries: approximately 0.7 seconds on average

These are observations from a small local test and are environment-dependent. They should not be interpreted as production performance benchmarks.

---

# LLM Usage & Estimated Cost

For each query, the application records:

* Prompt tokens

* Completion tokens

* Total tokens

* Cached input tokens when available

* Estimated API cost in USD when pricing is configured for the selected model
  These values are logged together with retrieved chunk count and query latency to provide basic inference-cost and performance observability.

Example local log:

Query completed: retrieved_chunks=3 latency_ms=1519.02

prompt_tokens=589 completion_tokens=92 total_tokens=681

cached_tokens=0 estimated_cost_usd=0.00014355

# Logging

The application uses Python's standard logging module to record operational events such as:

* Document parsing and chunk creation
* Document ingestion
* Number of retrieved chunks
* Query completion
* Query latency
* `NOT_FOUND` responses

The implementation avoids intentionally logging:

* API keys
* Full document contents
* Retrieved document text
* User questions

---

# Local Setup

## Requirements

The project requires:

* Python 3.12+
* A Groq API key
* Internet access for model/API access during setup and inference

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-question-answering
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the public GitHub repository URL.

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create the environment file:

```bash
cp .env.example .env
```

Configure:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model
```

Do not commit `.env` or API credentials to Git.

## 5. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Running Tests

Run the complete automated test suite:

```bash
pytest -q
```

The current suite contains 45 tests covering:

* API endpoints
* Request validation
* Document parsing
* Chunking
* Embeddings
* Vector store operations
* Retrieval
* Prompt construction
* LLM behavior
* Out-of-document handling

Latest local result:

```text
46 passed, 2 warnings
```

The warnings are dependency deprecation warnings and do not represent failed tests.

---

# End-to-End Usage

## 1. Start the application

```bash
uvicorn app.main:app --reload
```

## 2. Upload a document

```bash
curl -X POST \
  -F "file=@data/example.txt" \
  http://127.0.0.1:8000/documents/upload
```

The system:

```text
Document
   ↓
Parse
   ↓
Chunk
   ↓
Embed
   ↓
Store in ChromaDB
```

## 3. Ask a question

```bash
curl -X POST \
  http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the validation F1 score?"}'
```

The system:

```text
Question
   ↓
Query Embedding
   ↓
Top-3 Retrieval
   ↓
Grounded Prompt
   ↓
Groq LLM
   ↓
Answer + Sources
```

## 4. Ask an unsupported question

```bash
curl -X POST \
  http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Who was the CEO of the company?"}'
```

Expected behavior:

```json
{
  "answer": "NOT_FOUND",
  "sources": []
}
```

---

# Project Structure

```text
rag-question-answering/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and API endpoints
│   ├── config.py            # Application configuration
│   ├── schemas.py           # Pydantic request/response models
│   ├── parser.py            # PDF/TXT parsing and text cleaning
│   ├── chunker.py           # Structure-aware document chunking
│   ├── embeddings.py        # Embedding generation
│   ├── vector_store.py      # Persistent ChromaDB operations
│   ├── retriever.py         # Semantic retrieval
│   ├── llm.py               # Groq LLM integration
│   ├── prompts.py           # Grounding prompt construction
│   └── rag_pipeline.py      # End-to-end RAG orchestration
│
├── tests/
│   ├── test_api.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_prompts.py
│   └── test_llm.py
│
├── evaluation/
│   ├── questions.json
│   └── evaluate_retrieval.py
│
├── data/
│   └── .gitkeep
│
├── chroma_db/
│   └── .gitkeep
|
├── docs/
|   └── Avineesh_RAG_One_Page_Professional.pdf
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Configuration

The primary RAG configuration is defined in `app/config.py`.

| Parameter | Current Value | Purpose |
|---|---|---|
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `chroma_dir` | `./chroma_db` | Persistent vector store location |
| `chunk_size` | `700` | Target chunk size in characters |
| `chunk_overlap` | `120` | Chunk overlap in characters |
| `top_k` | `3` | Number of chunks retrieved per query |
| `relevance_threshold` | `0.0` | Distance filtering disabled |
| `groq_model` | `.env` | Groq generation model |
| `groq_api_key` | `.env` | Groq API authentication |

The application uses environment variables for secrets and model configuration where appropriate.

---

# Design Philosophy

The implementation intentionally keeps the main RAG pipeline explicit instead of hiding the core mechanics behind a high-level framework.

The following components can therefore be inspected and evaluated independently:

* Parsing
* Text cleaning
* Chunking
* Embedding
* Vector storage
* Retrieval
* Prompt construction
* LLM generation

This separation also makes it easier to identify where retrieval failures occur and evaluate individual design decisions.

The project prioritizes retrieval quality, grounded generation, source attribution, testability, and explainable engineering decisions.

---

# What Works

The current implementation supports:

* PDF ingestion
* TXT ingestion
* Text extraction and cleaning
* Structure-aware chunking
* Semantic embeddings
* Persistent ChromaDB storage
* Top-K semantic retrieval
* Groq-based grounded generation
* Source chunk attribution
* Out-of-document question handling
* Pydantic request validation
* API-level error handling
* Operational logging
* Automated tests
* Retrieval evaluation

---

# What Is Not Implemented

The following capabilities are outside the current implementation:

* OCR for scanned PDFs
* Image understanding
* Specialized table extraction
* Hybrid lexical + semantic search
* Retrieval reranking
* Authentication
* Multi-user document isolation
* Frontend UI
* Production deployment
* Advanced document management APIs
* Large-scale retrieval benchmarking

These limitations are documented explicitly rather than implying capabilities that are not currently implemented.

---

# Future Improvements

Potential future improvements include:

### 1. Larger Evaluation Dataset

Expand the retrieval evaluation with more questions covering different document structures, terminology, and difficulty levels.

### 2. Hybrid Retrieval

Combine dense semantic retrieval with lexical retrieval for exact technical terms, identifiers, and keyword-heavy questions.

### 3. Retrieval Reranking

Introduce a cross-encoder or similar reranking model after initial vector retrieval to improve candidate ordering.

### 4. Improved Document Parsing

Add OCR and better handling for:

* Scanned PDFs
* Tables
* Multi-column layouts
* Complex document formatting

### 5. Metadata-Aware Retrieval

Use metadata such as:

* Document ID
* Filename
* Page
* Section

to support more targeted retrieval.

### 6. Answer-Level Evaluation

Extend the current retrieval evaluation with metrics for:

* Answer correctness
* Groundedness
* Faithfulness
* Source relevance

### 7. Production Readiness

If the project moves beyond the local assignment, potential additions include:

* Containerization
* Hosted vector storage
* Authentication
* Monitoring
* Scalable model serving
* Multi-user document management

---

# Scope

This project focuses on demonstrating a working local RAG question-answering system rather than building a production-ready document platform.

The implementation prioritizes:

* Retrieval quality
* Grounded generation
* Source attribution
* Explainable design decisions
* Testability
* Clear separation of components
* Explicit documentation of limitations
