# OpenMCS-Agent
The agent of OpenMCS (Open Microscropy Control Software).

## RAG Enhancements
- Persistent vector store is saved under `resources/vector_store` (Chroma).
- File indexing tool updates the store when files change and tracks metadata.
- Retrieval pipeline scores relevance, rewrites low-quality queries, and answers.

### Install dependencies
```bash
pip install chromadb langchain-openai langchain-community langchain-text-splitters
```

### Configure embeddings
Set `Available embedding model` to use `text-embedding-v4` in `api_keys.yaml`.

### Tools
- `update_knowledge_base_from_files`: comma/semicolon-separated paths to files or directories.
- `search_knowledge_base`: retrieves top results with relevance scores.
- `rag_answer`: runs the full RAG pipeline and returns the final answer.
