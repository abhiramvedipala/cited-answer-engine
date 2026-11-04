import os

import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id         text PRIMARY KEY,
    config     text NOT NULL,
    doc        text NOT NULL,
    page_start int  NOT NULL,
    page_end   int  NOT NULL,
    chunk_idx  int  NOT NULL,
    text       text NOT NULL,
    embedding  vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_config ON chunks (config);
"""


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
