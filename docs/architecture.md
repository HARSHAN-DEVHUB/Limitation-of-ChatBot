# Architecture Notes

This implementation follows an offline Retrieval-Augmented Generation architecture:

1. Ingest public transcript-style datasets into normalized turn records.
2. Chunk conversation windows and store in SQLite.
3. Embed chunks using a local embedding model.
4. Index vectors with FAISS.
5. Retrieve top-k evidence during chat.
6. Generate grounded answers with explicit citations.

## Current MVP scope

- CLI chatbot
- SQLite storage
- Local embedding + FAISS retrieval
- Evaluation runner with baseline rubric metrics
