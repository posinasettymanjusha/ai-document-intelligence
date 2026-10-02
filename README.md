# AI Document Intelligence & RAG Platform

A full-stack platform foundation for managing workspaces and interacting with documents using AI. Phase 7 adds bounded grounded summaries, fixed-profile key-information extraction, and two-version comparisons. Embeddings are not generated during upload or analysis requests; reranking and hybrid search are not implemented.

## Technology Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS
- Backend: Python, FastAPI, Pydantic Settings
- Document extraction: PyMuPDF (PDF), python-docx (DOCX), Python standard library (TXT)
- Database: Supabase PostgreSQL, accessed through SQLAlchemy, Psycopg, and Alembic migrations
- Vectors: pgvector with a partial HNSW cosine index for ready embeddings
- File storage: private Supabase Storage bucket behind a storage adapter
- Identity: Supabase Auth; frontend client uses the public anon key, while backend access tokens are verified against the project's JWKS
- Tests: pytest and FastAPI TestClient

## Project Structure

```text
backend/
  app/
    api/v1/routes/       Versioned health, document, search, and answer endpoints
    auth/                Supabase token verification and request identity
    core/                Settings, logging, and API error handling
    db/                  PostgreSQL session, ORM models, and declarative base
    documents/           Processing, persistence orchestration, repository, and schemas
    chunking/            Provider-independent normalization and deterministic chunking
    embeddings/          Provider contract, generation service, validation, and status
    integrations/        Document extractors, private storage, and Gemini adapter
    workspaces/          Workspace schemas and service/repository boundaries
    main.py              FastAPI application factory and middleware
  tests/                 Backend tests
  pyproject.toml         Backend dependencies and pytest configuration
  alembic.ini            Migration configuration
  migrations/            Versioned PostgreSQL schema migrations, including document_chunks/vectors
  .env.example           Backend environment template
frontend/
  src/
    lib/                 Environment, API, and Supabase client setup
    App.tsx              Initial application shell and API health indicator
    main.tsx             React entry point
  package.json           Frontend dependencies and scripts
  vite.config.ts         Vite and Tailwind configuration
  .env.example           Frontend environment template
```

## Local Setup

Prerequisites: Node.js 20.19+ (or 22.12+), Python 3.11+, and a Supabase project or compatible local Supabase stack for persistence. Tests use in-memory repository/storage fakes and do not need production credentials.

### Backend

```powershell
cd backend
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Copy-Item .env.example .env
pip install -e ".[dev]"
```

Copy `backend/.env.example` to `backend/.env`, then set `DATABASE_URL`, `SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY`. Use a Supabase PostgreSQL connection URL with the `postgresql+psycopg://` scheme. The migration creates the tables and ensures that the `documents` Storage bucket is private. `MAX_DOCUMENT_SIZE_BYTES` is optional and defaults to 10 MiB.

Apply the database migration after configuring `DATABASE_URL`:

```powershell
alembic upgrade head
```

Run the backend from `backend/`:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/api/v1/health`. Interactive API docs: `http://127.0.0.1:8000/docs`.

Run backend tests from `backend/`:

```powershell
pytest
```

## Document Persistence and Chunking (Phases 2B-2C)

The upload flow validates a multipart file, verifies workspace membership, creates a document/version record with a SHA-256 checksum, stores the bytes in a private Supabase Storage bucket, extracts text, normalizes and chunks the segments, persists extracted segments and chunks with source metadata, and updates processing status. Re-uploading the same name and bytes returns the existing document; changed bytes create a new version. Deletion first tombstones the document, removes stored version objects, then removes database records so failed storage cleanup can be retried. A document or version scoped by a conversation cannot be deleted until that conversation is deleted; the API returns 409 before storage cleanup.

The schema has `profiles` (mirroring Supabase Auth users), `workspaces`, `workspace_members`, `documents`, `document_versions`, `document_chunks`, `conversations`, and `conversation_messages`. Conversations are private to an owner within a workspace, may have a fixed document/version scope, and use membership checks on every request. Messages have per-conversation sequence numbers and server-validated assistant citation snapshots in JSONB. `0003_document_embeddings` enables the PostgreSQL `vector` extension and adds nullable embedding data, model ID, dimension, status, and generated timestamp to chunks. `0004_embedding_failure_details` adds a bounded error message. `0005_conversations` adds conversation persistence and its ownership, scope, ordering, and cascade constraints. A partial HNSW cosine index covers only ready vectors. A migration trigger keeps profile rows aligned with Supabase Auth users.

Chunking is independent of routes, database implementation, and AI providers. Defaults are 1,200 characters per chunk, 150 characters of overlap, and a 100-character minimum split size. Whitespace is normalized within segments while meaningful paragraph boundaries are retained; boundaries prefer paragraphs, then whitespace, and never split words. Token counts use the deterministic character estimate `ceil(characters / 4)`, not an embedding-model tokenizer. Chunk rows are replaced and committed with extracted content and the ready status for that document version.

Embedding calls depend on the `EmbeddingProvider` interface (`embed_text` and `embed_texts`). `GeminiEmbeddingProvider` isolates the `google-genai` SDK and calls `gemini-embedding-2` with the configured `output_dimensionality` (768 by default). `EmbeddingService` validates count, dimension, and finite values. The local deterministic hash provider remains the default and needs no credentials; select `EMBEDDING_PROVIDER=gemini` and set the backend-only `GEMINI_API_KEY` to enable real calls. Provider errors are reduced to safe status messages before persistence; the API key and raw provider exception text are never stored or returned.

Generation is explicitly invoked at `POST /api/v1/documents/{document_id}/versions/{version_id}/embeddings`; upload does not call the provider. The service claims bounded batches, marks them `processing`, calls the provider, validates the complete response, then persists each batch atomically as `ready`. Failed batches store a sanitized error and can be retried with `retry_failed: true`. Ready chunks are excluded from claims and protected from overwrite. Semantic search embeds the query through the configured provider and ranks ready chunks with pgvector cosine distance. Search results are limited to the authenticated user's workspace and the current embedding model; the default local hash provider is deterministic plumbing, not a semantic model, so configure Gemini for meaningful semantic retrieval.

Grounded answers are available at `POST /api/v1/workspaces/{workspace_id}/answers`. The endpoint reuses semantic search, retrieves up to 5 chunks by default (request limit 1–10), and sends only those labeled excerpts to Gemini. Gemini is instructed to treat document text as untrusted evidence, answer only from that evidence, and cite factual claims with source labels. The API maps those labels to structured document, version, chunk, page, and source metadata; unknown labels are not returned. When search finds no chunks, or generation reports insufficient evidence, the API returns an explicit insufficient-context response without fabricated citations. Answer generation uses the server-side `GEMINI_API_KEY`, `GEMINI_GENERATION_MODEL` (`gemini-3.8-flash` by default), and `GEMINI_GENERATION_TIMEOUT_SECONDS` (30 by default). The generation model is separate from `EMBEDDING_MODEL`; answers from this endpoint are not stored.

Conversational document chat is available through workspace-scoped conversation endpoints. Conversations belong to one owner and one workspace, with optional immutable document/version scope. The server loads recent completed turns from PostgreSQL, bounds history by configured turn and character limits, and supplies it separately from retrieved excerpts. Prior assistant messages help resolve follow-ups but are not evidence; only current retrieved chunks support answers and citations. User messages are saved before retrieval/generation; failed turns remain visible with a failed status, and no assistant error message is stored. Concurrent active turns return 409. Settings default to 8 recent turns, 12,000 history characters, 8,000 question characters, and a 300-second stale-turn age. Automatic conversation summaries are not generated.

Document intelligence operations analyze ready document versions synchronously using ordered persisted chunks and source metadata. Summaries process the whole version in bounded map/reduce batches and cite original chunks. Key-information extraction accepts only the fixed `document_overview` profile and returns `found`, `missing`, or `ambiguous` fields with server-mapped citations. Comparisons authorize both versions first, analyze their chunks independently, and keep A/B source labels separate. Document text is untrusted evidence, never instructions. Hard limits cover chunks, estimated source/provider input tokens, provider calls, operation duration, output items, and output tokens; exceeding a limit fails explicitly rather than sampling a document. Results are not persisted and no migration is required.

Supabase Storage is accessed through an `ObjectStorage` integration using a backend-only service-role key. The bucket is private; the API does not return public object URLs. Tests override the repository and storage adapter with in-memory fakes, so live Supabase credentials are not required for the test suite.

API endpoints (each requires a valid Supabase Auth bearer token):

- `POST /api/v1/workspaces/{workspace_id}/documents` uploads and persists a document/version.
- `GET /api/v1/workspaces/{workspace_id}/documents` lists active workspace documents.
- `GET /api/v1/documents/{document_id}` returns the document and its version extraction data.
- `DELETE /api/v1/documents/{document_id}` removes the document and private file versions.
- `POST /api/v1/documents/{document_id}/versions/{version_id}/embeddings` explicitly generates pending chunk embeddings; set `retry_failed` to retry failed chunks.
- `POST /api/v1/workspaces/{workspace_id}/search` searches ready chunks in a workspace; provide `query`, optional `top_k` (1-50, default 10), and optional `document_id` and `version_id` filters.
- `POST /api/v1/workspaces/{workspace_id}/answers` returns a grounded answer and structured citations; provide `question`, optional `top_k` (1-10, default 5), and optional `document_id` and `version_id` filters.
- `POST /api/v1/workspaces/{workspace_id}/conversations` creates an owner-private conversation; optionally provide a title and document/version scope.
- `GET /api/v1/workspaces/{workspace_id}/conversations` lists the authenticated owner's conversations with cursor pagination.
- `GET /api/v1/workspaces/{workspace_id}/conversations/{conversation_id}` returns conversation metadata and an ordered, paginated message page.
- `POST /api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages` submits a question and returns its persisted user/assistant message pair with citations.
- `DELETE /api/v1/workspaces/{workspace_id}/conversations/{conversation_id}` deletes the private conversation and cascades to its messages.
- `POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{version_id}/summary` returns a grounded, cited summary; `style` is `executive` or `detailed`, with bounded `max_sections`.
- `POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{version_id}/key-information` returns the fixed `document_overview` extraction profile with explicit states and citations.
- `POST /api/v1/workspaces/{workspace_id}/document-comparisons` compares `document_a_id`/`version_a_id` with `document_b_id`/`version_b_id`; identical version pairs are rejected.

PDF extraction preserves 1-based page numbers, DOCX extraction preserves paragraph order/style and table-cell location, and TXT extraction preserves line ordering. Scanned PDFs without a text layer are rejected; TXT input must be UTF-8 (an optional UTF-8 BOM is accepted).

### Frontend

In a second PowerShell terminal, from the project root:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). `VITE_API_BASE_URL` defaults to `http://localhost:8000/api/v1`. Supabase frontend values are optional until authentication UI is added.

## Environment Variables

Backend variables are documented in `backend/.env.example`: `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, optional upload/chunk settings, `EMBEDDING_DIMENSION` (768), `EMBEDDING_PROVIDER` (local by default), `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE`, server-only `GEMINI_API_KEY`, `GEMINI_GENERATION_MODEL`, `GEMINI_GENERATION_TIMEOUT_SECONDS`, conversation limits, and document-analysis hard limits `DOCUMENT_ANALYSIS_MAX_CHUNKS`, `DOCUMENT_ANALYSIS_MAX_ESTIMATED_INPUT_TOKENS`, `DOCUMENT_ANALYSIS_BATCH_ESTIMATED_TOKENS`, `DOCUMENT_ANALYSIS_MAX_PROVIDER_CALLS`, `DOCUMENT_ANALYSIS_MAX_DURATION_SECONDS`, `DOCUMENT_ANALYSIS_MAX_OUTPUT_ITEMS`, and `DOCUMENT_ANALYSIS_MAX_OUTPUT_TOKENS`. Never put either service-role or Gemini keys in frontend variables or client code.

Frontend variables are documented in `frontend/.env.example`: `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY`. Vite exposes `VITE_` variables to browser code; never put a database password, Supabase service-role key, or AI provider secret in them.

## Current Scope

This includes the Phase 1 foundation, Phase 2B persistence, Phase 2C chunking, Phase 3A vector infrastructure, Phase 3B explicit embedding generation, Phase 4 semantic vector search, Phase 5 grounded document answers, Phase 6 persistent conversational document chat, and Phase 7 document intelligence. Embeddings run synchronously only when explicitly requested, with query embeddings generated for search, answers, and chat. Automatic conversation summaries, shared conversations, streaming, reranking, hybrid search, and analytics remain out of scope.