# Legal Document RAG Pipeline

A FastAPI-based RAG (Retrieval-Augmented Generation) pipeline for legal documents using the latest 2025 integrations: **LangChain + Qdrant + Sentence Transformers**.

## Features

- **Document Ingestion**: Upload and process PDF legal documents using LangChain
- **Semantic Search**: Query documents using Sentence Transformers embeddings
- **Vector Storage**: Store embeddings in Qdrant cloud with LangChain integration
- **Document Attribution**: Get precise source references for answers
- **FastAPI**: RESTful API with automatic documentation

## Technology Stack (2025)

- **Framework**: LangChain for document processing and embeddings
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: Qdrant Cloud via langchain-qdrant
- **Backend**: FastAPI
- **Document Processing**: LangChain PyPDFLoader with text splitting

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python run.py
   ```

   The API will be available at `http://localhost:8000`
   API documentation at `http://localhost:8000/docs`

## API Endpoints

### Health Check
```http
GET /health
```
Returns system status and collection information.

### Document Ingestion
```http
POST /ingest
```
Upload a PDF document for processing and storage.

**Parameters:**
- `file`: PDF file upload

**Response:**
```json
{
  "message": "Document ingested successfully",
  "document_name": "contract.pdf",
  "chunks_processed": 15,
  "status": "success"
}
```

### Query Documents
```http
POST /query
```
Query stored documents using natural language.

**Request Body:**
```json
{
  "question": "What are the termination procedures?",
  "k": 5
}
```

**Response:**
```json
{
  "answer": "Based on the relevant documents...",
  "sources": [
    {
      "document": "employment_contract.pdf",
      "chunk": 12,
      "page": 3,
      "confidence": 0.89,
      "text_excerpt": "The employee may be terminated..."
    }
  ],
  "query_time": "2025-09-29T21:26:29Z",
  "total_results": 3
}
```

### Collections Info
```http
GET /collections
```
Get information about stored document collections.

## Testing the Pipeline

### 1. Start the Server
```bash
python run.py
```

### 2. Test Health Endpoint
```bash
curl http://localhost:8000/health
```

### 3. Upload a Document
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### 4. Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

### 5. Run Automated Tests
```bash
python test_api.py
```

## Architecture

- **Document Processing**: LangChain PyPDFLoader with RecursiveCharacterTextSplitter
- **Text Chunking**: 500 characters with 50-character overlap
- **Embeddings**: Sentence Transformers via LangChain integration
- **Vector Database**: Qdrant Cloud with langchain-qdrant package
- **API Framework**: FastAPI with automatic documentation

## Configuration

The system uses Qdrant cloud credentials configured in `app/rag_service.py`. The service automatically:
- Creates collections if they don't exist
- Handles document chunking and metadata
- Provides similarity search with scores

## Sample Test Queries

After uploading legal documents, try these queries:
- "What are the termination procedures?"
- "How is confidential information protected?"
- "What are the payment obligations?"
- "Who is responsible for intellectual property?"

## Troubleshooting

1. **Connection Issues**: Check the `/health` endpoint first
2. **Upload Errors**: Ensure PDF files are valid and not password-protected
3. **Query Issues**: Try simpler queries if results are not relevant
4. **Dependencies**: Run `pip install -r requirements.txt` if imports fail

## Key Improvements in 2025 Version

- **LangChain Integration**: Native support for document processing and embeddings
- **Modern Dependencies**: Latest compatible versions of all libraries
- **Better Chunking**: RecursiveCharacterTextSplitter for smarter text splitting
- **Improved Metadata**: Rich document metadata through LangChain
- **Cleaner Architecture**: Separation of concerns with dedicated RAG service