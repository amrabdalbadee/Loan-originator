-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store PDF document metadata
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    total_pages INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table to store text chunks with their vector embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding vector(3072),
    created_at TIMESTAMP DEFAULT NOW()
);
