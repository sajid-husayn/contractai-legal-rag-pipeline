"""
FastAPI application for Legal Document RAG Pipeline
Using 2025 LangChain + Qdrant + Sentence Transformers integration
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

from .rag_service import rag_service
from .enhanced_rag_service import enhanced_rag_service
from .docling_rag_service import docling_rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Document RAG Pipeline",
    version="2.0.0",
    description="RAG pipeline using LangChain + Qdrant + Sentence Transformers"
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
    k: Optional[int] = 5

class Source(BaseModel):
    document: str
    chunk: int
    page: int
    confidence: float
    text_excerpt: str

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
    sources: List[Source]
    query_time: str
    total_results: int

class IngestResponse(BaseModel):
    message: str
    document_name: str
    chunks_processed: int
    status: str

class HierarchicalIngestResponse(BaseModel):
    message: str
    document_name: str
    chunks_processed: int
    status: str
    chunk_types: Optional[Dict[str, int]] = None
    section_categories: Optional[Dict[str, int]] = None

class HierarchicalQueryRequest(BaseModel):
    question: str
    k: Optional[int] = 10
    section_filter: Optional[str] = None
    level_filter: Optional[int] = None

class HierarchicalQueryResponse(BaseModel):
    answer: str
    sources: List[HierarchicalSource]
    query_time: str
    total_results: int
    approach_used: str
    filters_applied: Optional[Dict[str, Any]] = None

class QueryIntentResponse(BaseModel):
    query: str
    analysis: Dict[str, Any]
    recommendations: Dict[str, Any]
    expansion_queries: List[str]
    timestamp: str

class EnhancedQueryResponse(BaseModel):
    answer: str
    approach_used: str
    sources: List[Source]
    query_time: str
    total_results: int
    synthesis_used: Optional[bool] = None
    model: Optional[str] = None
    intent_analysis: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the basic UI"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/enhanced", response_class=HTMLResponse)
async def enhanced_ui():
    """Serve the enhanced comparison UI"""
    with open("static/index_enhanced.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/docling", response_class=HTMLResponse)
async def docling_comparison_ui():
    """Serve the 4-method comparison UI with Docling hierarchical approach"""
    with open("static/index_docling.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/health")
async def health():
    """Health check with collection information"""
    try:
        collection_info = rag_service.get_collection_info()
        return {
            "status": "healthy",
            "framework": "LangChain + Qdrant + Sentence Transformers",
            "collection": collection_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest PDF document using LangChain integration"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process with RAG service
        result = rag_service.ingest_document(content, file.filename)
        
        return IngestResponse(**result)
    
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using semantic search"""
    try:
        # Query with RAG service
        search_results = rag_service.query_documents(request.question, k=request.k)
        
        if not search_results:
            return QueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                query_time=datetime.now().isoformat(),
                total_results=0
            )
        
        # Format sources
        sources = []
        answer_parts = []
        
        for result in search_results:
            if result["confidence"] > 0.5:  # Similarity threshold
                source = Source(
                    document=result["document"],
                    chunk=result["chunk"],
                    page=result["page"],
                    confidence=round(result["confidence"], 3),
                    text_excerpt=result["text_excerpt"]
                )
                sources.append(source)
                answer_parts.append(result["full_content"][:300])
        
        # Create answer from relevant chunks
        if answer_parts:
            answer = "Based on the relevant documents:\n\n" + "\n\n---\n\n".join(answer_parts[:2])
        else:
            answer = "No sufficiently relevant information found for your query."
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time=datetime.now().isoformat(),
            total_results=len(search_results)
        )
    
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/collections")
async def get_collections():
    """Get information about stored collections"""
    try:
        return rag_service.get_collection_info()
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")

@app.post("/analyze_intent", response_model=QueryIntentResponse)
async def analyze_query_intent(request: QueryRequest):
    """Analyze query intent and provide recommendations"""
    try:
        result = enhanced_rag_service.analyze_query_intent(request.question)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryIntentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing query intent: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing intent: {str(e)}")

@app.post("/query_with_intent", response_model=EnhancedQueryResponse)
async def query_with_intent_understanding(request: QueryRequest):
    """Enhanced query using intent understanding and adaptive retrieval"""
    try:
        result = enhanced_rag_service.query_with_intent_understanding(
            request.question, 
            k=request.k
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Convert sources to Source objects
        sources = []
        for source_data in result.get("sources", []):
            sources.append(Source(**source_data))
        
        return EnhancedQueryResponse(
            answer=result["answer"],
            approach_used=result["approach_used"],
            sources=sources,
            query_time=result["query_time"], 
            total_results=result["total_results"],
            intent_analysis=result.get("intent_analysis")
        )
        
    except Exception as e:
        logger.error(f"Error in intent-based query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query_with_llm", response_model=EnhancedQueryResponse)
async def query_with_llm_synthesis(request: QueryRequest):
    """Enhanced query with LLM synthesis using Groq API"""
    try:
        # Use larger k for LLM synthesis to give more context
        enhanced_k = max(request.k * 2, 10)
        
        result = enhanced_rag_service.query_with_llm_synthesis(
            request.question,
            k=enhanced_k
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Convert sources to Source objects
        sources = []
        for source_data in result.get("sources", []):
            sources.append(Source(**source_data))
        
        return EnhancedQueryResponse(
            answer=result["answer"],
            approach_used=result["approach_used"],
            sources=sources,
            query_time=result["query_time"],
            total_results=result["total_results"],
            synthesis_used=result.get("synthesis_used"),
            model=result.get("model")
        )
        
    except Exception as e:
        logger.error(f"Error in LLM synthesis query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Hierarchical/Docling endpoints
@app.post("/ingest_hierarchical", response_model=HierarchicalIngestResponse)
async def ingest_document_hierarchical(file: UploadFile = File(...)):
    """Ingest PDF document using Docling with hierarchical structure extraction"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process with hierarchical Docling service
        result = docling_rag_service.ingest_document_hierarchical(content, file.filename)
        
        return HierarchicalIngestResponse(**result)
    
    except Exception as e:
        logger.error(f"Error ingesting document hierarchically: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query_hierarchical", response_model=HierarchicalQueryResponse)
async def query_documents_hierarchical(request: HierarchicalQueryRequest):
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
            return HierarchicalQueryResponse(
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
            answer = f"Based on hierarchical analysis of {len(set(s.document for s in sources))} documents:\n\n"
            
            # Group by section category for better organization
            by_category = {}
            for source in sources[:5]:  # Top 5 results
                category = source.section_category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(source)
            
            for category, category_sources in by_category.items():
                if category != "general":
                    answer += f"**{category.title()} Information:**\n"
                for source in category_sources[:2]:  # Top 2 per category
                    answer += f"- {source.document} ({source.chunk_type}, Level {source.level}): {source.text_excerpt}...\n"
                answer += "\n"
        else:
            answer = "No sufficiently relevant information found for your query."
        
        return HierarchicalQueryResponse(
            answer=answer,
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
        logger.error(f"Error querying documents hierarchically: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/collections_hierarchical")
async def get_hierarchical_collections():
    """Get information about hierarchical document collections"""
    try:
        return docling_rag_service.get_collection_info()
    except Exception as e:
        logger.error(f"Error getting hierarchical collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")

@app.get("/health_hierarchical")
async def health_hierarchical():
    """Health check for hierarchical collection"""
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)