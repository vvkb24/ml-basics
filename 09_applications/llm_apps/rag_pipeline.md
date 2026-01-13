# RAG Pipeline

> **Status:** Stub - Implementation coming soon

## Overview

Retrieval-Augmented Generation combines retrieval with LLM generation.

## Components

1. **Document Store**: Vector database (FAISS, Pinecone)
2. **Embeddings**: Sentence transformers
3. **Retriever**: Semantic search
4. **Generator**: LLM for response

## Pipeline

```
Query → Embed → Retrieve → Rerank → Generate → Response
```

## Implementation Topics

- Chunking strategies
- Embedding models
- Vector similarity search
- Prompt engineering
- Answer faithfulness
