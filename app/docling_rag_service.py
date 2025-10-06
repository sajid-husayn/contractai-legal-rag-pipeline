"""
Enhanced RAG service using Docling for hierarchical document structure extraction
Implements multi-level embedding strategy for legal documents
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import io

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import DoclingDocument, NodeItem

from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

class HierarchicalChunk:
    """Represents a hierarchical chunk with multiple levels of context"""
    
    def __init__(
        self,
        content: str,
        chunk_type: str,  # 'document', 'section', 'subsection', 'paragraph'
        level: int,
        title: str = "",
        parent_title: str = "",
        document_name: str = "",
        page_number: int = 0,
        metadata: Dict[str, Any] = None
    ):
        self.content = content
        self.chunk_type = chunk_type
        self.level = level
        self.title = title
        self.parent_title = parent_title
        self.document_name = document_name
        self.page_number = page_number
        self.metadata = metadata or {}
        self.chunk_id = f"{document_name}_{chunk_type}_{level}_{hash(content) % 10000}"

class DoclingRAGService:
    """Enhanced RAG service using Docling for structure-aware document processing"""
    
    def __init__(self, collection_name: str = "legal_documents_hierarchical"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Qdrant Cloud client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "https://e73af034-4c6c-429a-8c81-2168a43033f2.us-west-2-0.aws.cloud.qdrant.io:6333"),
            api_key=os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._o0OW9Sx4DI0MJIB7la5yj4Z6gfndYW5y97YfZ-5jI4")
        )
        
        # Initialize Docling converter
        self.converter = self._setup_docling_converter()
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Legal document section patterns
        self.legal_sections = {
            "payment": ["payment", "compensation", "salary", "fee", "billing", "invoice", "remuneration"],
            "termination": ["termination", "terminate", "end", "cancel", "dissolution", "expiry"],
            "confidentiality": ["confidential", "nda", "non-disclosure", "secret", "proprietary"],
            "intellectual_property": ["intellectual property", "ip", "copyright", "patent", "work product", "invention"],
            "liability": ["liability", "damages", "indemnification", "responsible", "accountable"],
            "scope": ["scope", "services", "deliverables", "work", "duties", "responsibilities"],
            "general": ["whereas", "agreement", "parties", "effective date", "governing law"]
        }
    
    def _setup_docling_converter(self) -> DocumentConverter:
        """Setup Docling document converter with legal document optimizations"""
        try:
            # Configure pipeline for legal documents
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned documents
            pipeline_options.do_table_structure = True  # Extract table structure
            pipeline_options.table_structure_options.do_cell_matching = True
            
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            logger.info("Docling converter initialized successfully")
            return doc_converter
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the hierarchical collection exists"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def process_document_with_docling(self, pdf_content: bytes, filename: str) -> DoclingDocument:
        """Process PDF using Docling to extract structure"""
        try:
            # Create temporary file since Docling expects file path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file.flush()
                
                # Convert document using file path
                result = self.converter.convert(temp_file.name)
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
            logger.info(f"Successfully processed {filename} with Docling")
            return result.document
            
        except Exception as e:
            logger.error(f"Error processing document with Docling: {e}")
            raise
    
    def extract_hierarchical_chunks(self, docling_doc: DoclingDocument, filename: str) -> List[HierarchicalChunk]:
        """Extract hierarchical chunks from Docling document"""
        chunks = []
        
        try:
            # Get the full text content from Docling
            full_text = docling_doc.export_to_markdown()
            
            # Split the content into meaningful chunks using text splitting
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", ".", " ", ""]
            )
            
            # Split the full text into chunks
            text_chunks = text_splitter.split_text(full_text)
            
            current_section = ""
            
            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue
                
                # Determine section category for legal documents
                section_category = self._categorize_legal_section(chunk_text)
                
                # Extract section title if this looks like a heading
                title = self._extract_title_from_chunk(chunk_text)
                if title and len(title) < 100:
                    current_section = title
                
                # Determine chunk type based on content
                chunk_type, level = self._determine_chunk_type(chunk_text, i)
                
                # Create hierarchical chunk
                chunk = HierarchicalChunk(
                    content=chunk_text,
                    chunk_type=chunk_type,
                    level=level,
                    title=title or chunk_text[:50],
                    parent_title=current_section,
                    document_name=filename,
                    page_number=i // 3,  # Rough page estimation
                    metadata={
                        "section_category": section_category,
                        "parent_section": current_section,
                        "chunk_index": i,
                        "content_length": len(chunk_text)
                    }
                )
                
                chunks.append(chunk)
            
            logger.info(f"Extracted {len(chunks)} hierarchical chunks from {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting hierarchical chunks: {e}")
            return []
    
    def _extract_title_from_chunk(self, text: str) -> str:
        """Extract title from chunk if it looks like a heading"""
        lines = text.strip().split('\n')
        first_line = lines[0].strip()
        
        # Check if first line looks like a heading
        if len(first_line) < 100 and (
            first_line.isupper() or 
            first_line.startswith('#') or
            any(keyword in first_line.lower() for keyword in ["article", "section", "clause", "whereas"])
        ):
            return first_line.replace('#', '').strip()
        
        return ""
    
    def _determine_chunk_type(self, text: str, index: int) -> tuple:
        """Determine chunk type and level based on content"""
        text_lower = text.lower()
        
        # Check for section headers
        if any(keyword in text_lower[:100] for keyword in ["article", "section", "clause"]):
            return "section", 1
        
        # Check for subsection patterns
        if text.strip().startswith(("a)", "b)", "c)", "1.", "2.", "3.", "i.", "ii.", "iii.")):
            return "subsection", 2
        
        # Default to paragraph
        return "paragraph", 3
    
    def _classify_content_chunk(self, item: NodeItem, text: str) -> Dict[str, Any]:
        """Classify content chunk type and level based on Docling item"""
        item_type = type(item).__name__.lower()
        
        # Check if it's a heading/title
        if any(heading_word in text.lower()[:50] for heading_word in ["article", "section", "clause", "whereas"]):
            if len(text) < 200:  # Likely a section header
                return {
                    "chunk_type": "section",
                    "level": 1,
                    "title": text[:100]
                }
        
        # Check for subsection patterns
        if text.startswith(("a)", "b)", "c)", "1.", "2.", "3.", "i.", "ii.", "iii.")):
            return {
                "chunk_type": "subsection", 
                "level": 2,
                "title": text[:50]
            }
        
        # Default to paragraph
        return {
            "chunk_type": "paragraph",
            "level": 3, 
            "title": text[:50]
        }
    
    def _categorize_legal_section(self, text: str) -> str:
        """Categorize text into legal section types"""
        text_lower = text.lower()
        
        for category, keywords in self.legal_sections.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "general"
    
    def ingest_document_hierarchical(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Ingest document using hierarchical chunking approach"""
        try:
            # Process with Docling
            docling_doc = self.process_document_with_docling(pdf_content, filename)
            
            # Extract hierarchical chunks
            chunks = self.extract_hierarchical_chunks(docling_doc, filename)
            
            if not chunks:
                return {
                    "message": "No content extracted from document",
                    "document_name": filename,
                    "chunks_processed": 0,
                    "status": "failed"
                }
            
            # Generate embeddings and store
            points = []
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embeddings.embed_query(chunk.content)
                    
                    # Create point for Qdrant
                    point = PointStruct(
                        id=hash(chunk.chunk_id) % (2**63 - 1),  # Ensure positive integer
                        vector=embedding,
                        payload={
                            "document": chunk.document_name,
                            "chunk_id": chunk.chunk_id,
                            "chunk_type": chunk.chunk_type,
                            "level": chunk.level,
                            "title": chunk.title,
                            "parent_title": chunk.parent_title,
                            "page": chunk.page_number,
                            "content": chunk.content,
                            "text_excerpt": chunk.content[:200],
                            "section_category": chunk.metadata.get("section_category", "general"),
                            "metadata": chunk.metadata,
                            "ingestion_time": datetime.now().isoformat()
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Batch insert into Qdrant
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            return {
                "message": f"Successfully processed {filename} with hierarchical chunking",
                "document_name": filename,
                "chunks_processed": len(points),
                "status": "success",
                "chunk_types": {chunk.chunk_type: sum(1 for c in chunks if c.chunk_type == chunk.chunk_type) for chunk in chunks},
                "section_categories": {chunk.metadata.get("section_category", "general"): sum(1 for c in chunks if c.metadata.get("section_category") == chunk.metadata.get("section_category")) for chunk in chunks}
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical document ingestion: {e}")
            return {
                "message": f"Error processing document: {str(e)}",
                "document_name": filename,
                "chunks_processed": 0,
                "status": "error"
            }
    
    def query_hierarchical(
        self, 
        query: str, 
        k: int = 10,
        section_filter: Optional[str] = None,
        level_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query with hierarchical structure awareness"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build filter conditions
            filter_conditions = {}
            if section_filter:
                filter_conditions["section_category"] = section_filter
            if level_filter is not None:
                filter_conditions["level"] = level_filter
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter={"must": [{"key": key, "match": {"value": value}} for key, value in filter_conditions.items()]} if filter_conditions else None,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                payload = result.payload
                results.append({
                    "document": payload["document"],
                    "chunk_id": payload["chunk_id"],
                    "chunk_type": payload["chunk_type"],
                    "level": payload["level"],
                    "title": payload["title"],
                    "parent_title": payload["parent_title"],
                    "page": payload["page"],
                    "confidence": float(result.score),
                    "text_excerpt": payload["text_excerpt"],
                    "full_content": payload["content"],
                    "section_category": payload["section_category"],
                    "metadata": payload["metadata"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hierarchical query: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the hierarchical collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get sample of chunk types and categories
            sample_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )[0]
            
            chunk_types = {}
            section_categories = {}
            documents = set()
            
            for point in sample_results:
                payload = point.payload
                chunk_type = payload.get("chunk_type", "unknown")
                section_category = payload.get("section_category", "general")
                document = payload.get("document", "unknown")
                
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                section_categories[section_category] = section_categories.get(section_category, 0) + 1
                documents.add(document)
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": collection_info.points_count,
                "documents_count": len(documents),
                "chunk_types": chunk_types,
                "section_categories": section_categories,
                "documents": list(documents),
                "embedding_dimension": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from the vector database collection"""
        try:
            # Delete the collection
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Recreate the collection
            self._ensure_collection_exists()
            logger.info(f"Recreated collection: {self.collection_name}")
            
            return {
                "message": f"Collection {self.collection_name} cleared and recreated successfully",
                "collection_name": self.collection_name,
                "status": "cleared"
            }
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return {"error": str(e)}

# Initialize the service
docling_rag_service = DoclingRAGService()