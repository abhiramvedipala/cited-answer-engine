FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 HF_HOME=/models
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Bake the local models into the image so the first request is not a download.
RUN uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY ingest ingest
COPY retrieval retrieval
COPY answer answer
COPY service service
COPY eval eval
COPY scripts scripts
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "--factory", "service.api:app_factory", "--host", "0.0.0.0", "--port", "8000"]
