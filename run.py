#!/usr/bin/env python3
import uvicorn

if __name__ == "__main__":
    print("Starting Legal Document RAG Pipeline...")
    print("Using: LangChain + Qdrant + HuggingFace Embeddings")
    print("API Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)