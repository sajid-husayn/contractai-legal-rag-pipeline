"""
RAG service using 2025 LangChain + Qdrant + HuggingFace Embeddings integration
Using all-MiniLM-L6-v2 model following official LangChain docs
"""
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from typing import List, Dict, Any
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        """Initialize RAG service with 2025 LangChain + Qdrant integration"""
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = "legal_documents"
        
        # Initialize embeddings with HuggingFace (following official docs)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, pdf_content: bytes, filename: str) -> List[Document]:
        """Process PDF using LangChain PyPDFLoader"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file_path = tmp_file.name
            
            # Load PDF with LangChain
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "document_name": filename,
                    "page_number": i + 1,
                    "chunk_index": i
                })
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Update chunk indices
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_id"] = f"{filename}_chunk_{i}"
            
            logger.info(f"Processed {filename}: {len(chunks)} chunks from {len(documents)} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise
    
    def ingest_document(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Ingest document using LangChain + Qdrant integration"""
        try:
            # Process PDF into chunks
            chunks = self.process_pdf(pdf_content, filename)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            return {
                "message": "Document ingested successfully",
                "document_name": filename,
                "chunks_processed": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    def query_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query documents using semantic search"""
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "document": doc.metadata.get("document_name", "unknown"),
                    "chunk": doc.metadata.get("chunk_index", 0),
                    "page": doc.metadata.get("page_number", 0),
                    "confidence": float(score),
                    "text_excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "full_content": doc.page_content
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"status": "error", "error": str(e)}

# Global instance
rag_service = RAGService(
    qdrant_url="https://e73af034-4c6c-429a-8c81-2168a43033f2.us-west-2-0.aws.cloud.qdrant.io:6333",
    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._o0OW9Sx4DI0MJIB7la5yj4Z6gfndYW5y97YfZ-5jI4"
)