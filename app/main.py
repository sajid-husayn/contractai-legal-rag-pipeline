"""
FastAPI application for Legal Document RAG Pipeline - Simplified Docling-only Version
Using Docling + Qdrant + Sentence Transformers integration
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from .docling_rag_service import docling_rag_service
from .llm_synthesis import grok_synthesis_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Document RAG Pipeline - Docling",
    version="3.0.0",
    description="Structure-Aware RAG pipeline using Docling + Qdrant + Sentence Transformers"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 10
    section_filter: Optional[str] = None
    level_filter: Optional[int] = None

class HierarchicalSource(BaseModel):
    document: str
    chunk_id: str
    chunk_type: str
    level: int
    title: str
    parent_title: str
    page: int
    confidence: float
    text_excerpt: str
    section_category: str

class QueryResponse(BaseModel):
    answer: str
    llm_answer: Optional[str] = None
    sources: List[HierarchicalSource]
    query_time: str
    total_results: int
    approach_used: str
    filters_applied: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    message: str
    document_name: str
    chunks_processed: int
    status: str
    chunk_types: Optional[Dict[str, int]] = None
    section_categories: Optional[Dict[str, int]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the simplified UI"""
    with open("static/index_simple.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/health")
async def health():
    """Health check with collection information"""
    try:
        collection_info = docling_rag_service.get_collection_info()
        return {
            "status": "healthy",
            "framework": "Docling + Qdrant + Sentence Transformers + Hierarchical Chunking",
            "collection": collection_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest PDF document using Docling with hierarchical structure extraction"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process with hierarchical Docling service
        result = docling_rag_service.ingest_document_hierarchical(content, file.filename)
        
        return IngestResponse(**result)
    
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Keep the hierarchical endpoints for backward compatibility
@app.post("/ingest_hierarchical", response_model=IngestResponse)
async def ingest_document_hierarchical(file: UploadFile = File(...)):
    """Ingest PDF document using Docling with hierarchical structure extraction"""
    return await ingest_document(file)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using hierarchical structure-aware search"""
    try:
        # Query with hierarchical service
        search_results = docling_rag_service.query_hierarchical(
            request.question, 
            k=request.k,
            section_filter=request.section_filter,
            level_filter=request.level_filter
        )
        
        if not search_results:
            return QueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                query_time=datetime.now().isoformat(),
                total_results=0,
                approach_used="hierarchical_docling",
                filters_applied={
                    "section_filter": request.section_filter,
                    "level_filter": request.level_filter
                }
            )
        
        # Format sources
        sources = []
        answer_parts = []
        
        for result in search_results:
            if result["confidence"] > 0.5:  # Similarity threshold
                source = HierarchicalSource(
                    document=result["document"],
                    chunk_id=result["chunk_id"],
                    chunk_type=result["chunk_type"],
                    level=result["level"],
                    title=result["title"],
                    parent_title=result["parent_title"],
                    page=result["page"],
                    confidence=round(result["confidence"], 3),
                    text_excerpt=result["text_excerpt"],
                    section_category=result["section_category"]
                )
                sources.append(source)
                answer_parts.append(result["full_content"][:300])
        
        # Create structured answer based on hierarchical information
        if answer_parts:
            answer = f"Based on hierarchical analysis of {len(set(s.document for s in sources))} documents:\\n\\n"
            
            # Group by section category for better organization
            by_category = {}
            for source in sources[:5]:  # Top 5 results
                category = source.section_category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(source)
            
            for category, category_sources in by_category.items():
                if category != "general":
                    answer += f"**{category.title()} Information:**\\n"
                for source in category_sources[:2]:  # Top 2 per category
                    answer += f"- {source.document} ({source.chunk_type}, Level {source.level}): {source.text_excerpt}...\\n"
                answer += "\\n"
        else:
            answer = "No sufficiently relevant information found for your query."
        
        # Generate LLM synthesis
        llm_answer = None
        if search_results and len(sources) > 0:
            try:
                llm_answer = grok_synthesis_service.synthesize_answer(request.question, search_results)
            except Exception as e:
                logger.error(f"Error in LLM synthesis: {e}")
                llm_answer = f"LLM synthesis error: {str(e)}"
        
        return QueryResponse(
            answer=answer,
            llm_answer=llm_answer,
            sources=sources,
            query_time=datetime.now().isoformat(),
            total_results=len(search_results),
            approach_used="hierarchical_docling",
            filters_applied={
                "section_filter": request.section_filter,
                "level_filter": request.level_filter
            }
        )
    
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query_hierarchical", response_model=QueryResponse)
async def query_documents_hierarchical(request: QueryRequest):
    """Query documents using hierarchical structure-aware search"""
    return await query_documents(request)

@app.delete("/clear-database")
async def clear_database():
    """Clear all documents from the vector database"""
    try:
        # Clear all points from the collection
        result = docling_rag_service.clear_collection()
        logger.info("Database cleared successfully")
        return {"message": "Database cleared successfully", "result": result}
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/collections")
async def get_collections():
    """Get information about stored collections"""
    try:
        return docling_rag_service.get_collection_info()
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")

# Keep hierarchical endpoints for backward compatibility
@app.get("/collections_hierarchical")
async def get_hierarchical_collections():
    """Get information about hierarchical document collections"""
    return await get_collections()

@app.get("/health_hierarchical")
async def health_hierarchical():
    """Health check for hierarchical collection"""
    return await health()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)