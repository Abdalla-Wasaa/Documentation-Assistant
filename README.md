# Documentation Assistant

A Python assistant that retrieves LangChain documentation from Pinecone and answers questions using OpenRouter. Documentation is crawled with Tavily, split into chunks, and embedded before indexing.

## Setup

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with your OpenRouter, Pinecone, and Tavily credentials and your Pinecone index name. Use an index compatible with the configured `openai/text-embedding-3-small` embeddings.

## Run

From the project root, ingest documentation into your Pinecone index:

```bash
uv run python src/documentation_helper/ingestion.py
```

Then run the backend's example question:

```bash
uv run python backend/core.py
```

The backend also exposes `run_llm(query)` for programmatic use. API calls may incur provider charges. Credentials, virtual environments, and local vector databases are excluded from new commits.
