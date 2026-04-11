-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store user details (applicants)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name TEXT,
    national_id TEXT UNIQUE,
    phone TEXT,
    email TEXT
);

-- Table to store PDF document metadata
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    total_pages INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table to store policy text chunks with their vector embeddings
CREATE TABLE IF NOT EXISTS policy_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding vector(3072),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table to store user-uploaded text chunks (raw text only, no embeddings)
CREATE TABLE IF NOT EXISTS user_documents (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
