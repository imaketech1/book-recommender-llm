# Book Recommender Backend (LLM-based)

This is a FastAPI backend for recommending books based on natural language queries, powered by SentenceTransformers and FAISS.

## Features

- Natural language query parsing
- Semantic search over book dataset
- Fast recommendations via FAISS

## Setup

```bash
git clone git@github.com:imaketech1/book-recommender-llm.git
cd book-recommender-llm
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🔗 Frontend Repository

You can find the frontend code here: [book-recommender-frontend](https://github.com/imaketech1/book-recommender-frontend)
