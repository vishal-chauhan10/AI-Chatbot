"""
Adhyatmik Intelligence AI - FastAPI Backend
A multilingual RAG chatbot backend with Context7 integration
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our new services
from services.vector_db import initialize_vector_db
from services.embedding_service import initialize_embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for ML models and resources
ml_models: Dict[str, any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Handles ML model loading and resource cleanup.
    
    What happens during startup:
    1. Initialize vector database (ChromaDB)
    2. Load embedding models for text-to-vector conversion
    3. Set up translation services
    4. Verify all systems are working
    """
    logger.info("🚀 Starting Adhyatmik Intelligence AI Backend...")
    
    # Initialize services in order of dependency
    try:
        # 1. Initialize embedding service (needed by vector DB)
        logger.info("📊 Initializing embedding service...")
        embedding_service = initialize_embedding_service()
        ml_models["embedding_service"] = embedding_service
        
        # 2. Initialize vector database
        logger.info("🗄️ Initializing vector database...")
        vector_db = initialize_vector_db()
        ml_models["vector_db"] = vector_db
        
        # 3. Set up other services
        ml_models["embedding_model"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ml_models["translation_service"] = "google_translate"  # TODO: Implement
        ml_models["language_detector"] = "langdetect"  # TODO: Implement
        
        # 4. Verify database connection
        stats = vector_db.get_collection_stats("transcripts")
        logger.info(f"📈 Database stats: {stats}")
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup resources
    logger.info("🔄 Shutting down Adhyatmik Intelligence AI Backend...")
    ml_models.clear()
    logger.info("✅ Resources cleaned up successfully")


# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="Adhyatmik Intelligence AI",
    description="A multilingual RAG chatbot backend supporting Gujarati, Hindi, English, and Hinglish",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    """Chat message model with comprehensive validation"""
    content: str = Field(..., min_length=1, max_length=2000, description="Message content")
    language: Optional[str] = Field(None, description="Detected or specified language")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(None, description="User identifier")


class ChatResponse(BaseModel):
    """Chat response model with metadata"""
    response: str = Field(..., description="Generated response")
    language: str = Field(..., description="Response language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="ML models status")


# Dependency for ML models
async def get_ml_models() -> Dict[str, any]:
    """
    Dependency to access loaded ML models.
    Ensures models are available before processing requests.
    """
    if not ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Please try again later."
        )
    return ml_models


# API Routes
@app.get("/", response_model=Dict[str, str])
async def read_root():
    """Root endpoint returning API information"""
    return {
        "message": "Welcome to Adhyatmik Intelligence AI",
        "description": "Multilingual RAG Chatbot Backend",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check(models: Dict[str, any] = Depends(get_ml_models)):
    """
    Health check endpoint for monitoring service status.
    Returns comprehensive system health information.
    """
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        models_loaded=bool(models)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    message: ChatMessage,
    models: Dict[str, any] = Depends(get_ml_models)
):
    """
    Main chat endpoint for processing multilingual queries.
    
    Supports:
    - Gujarati (original and romanized)
    - Hindi
    - English
    - Hinglish (Hindi-English mix)
    
    Process:
    1. Language detection
    2. Query translation (if needed)
    3. RAG retrieval from Context7
    4. Response generation
    5. Response translation (if needed)
    """
    start_time = datetime.utcnow()
    
    try:
        # TODO: Implement actual chat processing logic
        # This is a placeholder implementation
        
        # Language detection
        detected_language = message.language or "english"  # Placeholder
        
        # Mock response for now
        response_text = f"Thank you for your message: '{message.content}'. This is a placeholder response from Adhyatmik Intelligence AI."
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ChatResponse(
            response=response_text,
            language=detected_language,
            confidence=0.95,
            sources=["placeholder_source.txt"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your message: {str(e)}"
        )


@app.get("/languages", response_model=List[Dict[str, str]])
async def get_supported_languages():
    """Get list of supported languages"""
    return [
        {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
        {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
        {"code": "en", "name": "English", "native": "English"},
        {"code": "hi-en", "name": "Hinglish", "native": "Hinglish"}
    ]


@app.get("/stats", response_model=Dict[str, Union[int, str]])
async def get_system_stats(models: Dict[str, any] = Depends(get_ml_models)):
    """Get system statistics and model information"""
    # Get vector database stats
    vector_db = models.get("vector_db")
    db_stats = {}
    if vector_db:
        db_stats = vector_db.get_collection_stats("transcripts")
    
    return {
        "models_loaded": len(models),
        "embedding_model": models.get("embedding_model", "Not loaded"),
        "translation_service": models.get("translation_service", "Not loaded"),
        "language_detector": models.get("language_detector", "Not loaded"),
        "vector_db_documents": db_stats.get("document_count", 0),
        "vector_db_status": db_stats.get("status", "unknown"),
        "uptime": "System running",
        "status": "operational"
    }


# Pydantic model for document addition
class DocumentAddRequest(BaseModel):
    """Request model for adding documents to the vector database"""
    content: str = Field(..., description="Document content")
    document_id: str = Field(..., description="Unique document identifier")
    language: str = Field(default="gujarati", description="Document language")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

# New endpoints for vector database testing and management
@app.post("/documents/add")
async def add_document(
    request: DocumentAddRequest,
    models: Dict[str, any] = Depends(get_ml_models)
):
    """
    Add a document to the vector database.
    
    This endpoint allows you to:
    1. Upload session transcripts
    2. Store them with embeddings for semantic search
    3. Add metadata for better organization
    
    Perfect for testing our RAG system!
    """
    try:
        vector_db = models.get("vector_db")
        if not vector_db:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Prepare metadata
        doc_metadata = {
            "language": request.language,
            "timestamp": datetime.utcnow().isoformat(),
            "content_type": "session_transcript"
        }
        if request.metadata:
            doc_metadata.update(request.metadata)
        
        # Add document to vector database
        success = vector_db.add_document(
            content=request.content,
            document_id=request.document_id,
            metadata=doc_metadata,
            collection_name="transcripts"
        )
        
        if success:
            return {
                "message": "Document added successfully",
                "document_id": request.document_id,
                "language": request.language,
                "metadata": doc_metadata
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")
            
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for document search
class DocumentSearchRequest(BaseModel):
    """Request model for searching documents"""
    query: str = Field(..., description="Search query")
    language: Optional[str] = Field(default=None, description="Filter by language")
    max_results: int = Field(default=5, description="Maximum number of results")

@app.post("/documents/search")
async def search_documents(
    request: DocumentSearchRequest,
    models: Dict[str, any] = Depends(get_ml_models)
):
    """
    Search for documents using semantic similarity.
    
    This is the core of our RAG system:
    1. Takes your query in any language
    2. Finds semantically similar content
    3. Returns relevant documents with similarity scores
    
    Try queries like:
    - "What was discussed about meditation?"
    - "કોઈ આધ્યાત્મિક વિષય વિશે શું કહેવાયું?" (Gujarati)
    - "मेडिटेशन के बारे में क्या कहा गया?" (Hindi)
    """
    try:
        vector_db = models.get("vector_db")
        if not vector_db:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Prepare filter for language if specified
        where_filter = None
        if request.language:
            where_filter = {"language": request.language}
        
        # Search for similar documents
        results = vector_db.search_similar(
            query=request.query,
            collection_name="transcripts",
            n_results=request.max_results,
            where_filter=where_filter
        )
        
        return {
            "query": request.query,
            "language_filter": request.language,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats")
async def get_document_stats(models: Dict[str, any] = Depends(get_ml_models)):
    """Get statistics about stored documents"""
    try:
        vector_db = models.get("vector_db")
        if not vector_db:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Get stats for all collections
        transcripts_stats = vector_db.get_collection_stats("transcripts")
        chunks_stats = vector_db.get_collection_stats("chunks")
        
        return {
            "collections": {
                "transcripts": transcripts_stats,
                "chunks": chunks_stats
            },
            "total_documents": transcripts_stats.get("document_count", 0) + chunks_stats.get("document_count", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the server with optimal settings for development
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )