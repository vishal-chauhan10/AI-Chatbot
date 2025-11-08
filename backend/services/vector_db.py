"""
Vector Database Service using ChromaDB
=====================================

This module handles all vector database operations for our RAG system.

Key Concepts Explained:
1. **Vector Database**: Stores mathematical representations (embeddings) of text
2. **Collections**: Like tables in traditional databases, but for vectors
3. **Embeddings**: Numerical representations that capture semantic meaning
4. **Similarity Search**: Finding documents with similar meaning, not exact words

Why ChromaDB?
- Local-first: Runs on your machine, no external API calls needed
- Python-native: Built specifically for Python applications
- Persistent: Data survives server restarts
- Multilingual: Works well with different languages including Gujarati
"""

import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabaseService:
    """
    Manages ChromaDB operations for our multilingual RAG system.
    
    This class handles:
    - Database initialization and connection
    - Document storage with embeddings
    - Semantic search and retrieval
    - Collection management for different document types
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB client and setup collections.
        
        Args:
            persist_directory: Where to store the database files locally
            
        Why we need persistence:
        - Without it, all data is lost when the server restarts
        - Allows us to build up a knowledge base over time
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        # This creates a local database that survives server restarts
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True  # Allow database reset during development
            )
        )
        
        # Set up the embedding function
        # This converts text to vectors (mathematical representations)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Why this specific model?
        # - "paraphrase-multilingual": Works with multiple languages including Gujarati
        # - "mpnet": High-quality embeddings that capture semantic meaning well
        # - "base-v2": Good balance of performance and accuracy
        
        # Initialize collections for different types of content
        self.collections = {}
        self._initialize_collections()
        
        logger.info(f"✅ VectorDB initialized with persistence at: {self.persist_directory}")
    
    def _initialize_collections(self):
        """
        Create collections for different types of documents.
        
        Collections are like tables in traditional databases:
        - Each collection can have different embedding models
        - Allows us to organize content by type or language
        - Can have different search strategies per collection
        """
        try:
            # Main collection for session transcripts
            self.collections['transcripts'] = self.client.get_or_create_collection(
                name="session_transcripts",
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Gujarati session transcripts and their translations",
                    "languages": "gujarati,hindi,english",  # Convert list to comma-separated string
                    "content_type": "session_transcripts"
                }
            )
            
            # Collection for processed chunks (smaller pieces of documents)
            self.collections['chunks'] = self.client.get_or_create_collection(
                name="document_chunks",
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Processed text chunks for better retrieval",
                    "chunk_strategy": "semantic_with_overlap",
                    "max_chunk_size": "500"  # Convert int to string
                }
            )
            
            logger.info("✅ Collections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collections: {e}")
            raise
    
    def add_document(
        self, 
        content: str, 
        document_id: str, 
        metadata: Dict[str, Any],
        collection_name: str = "transcripts"
    ) -> bool:
        """
        Add a document to the vector database.
        
        Args:
            content: The actual text content
            document_id: Unique identifier for the document
            metadata: Additional information (language, date, speaker, etc.)
            collection_name: Which collection to store in
            
        Returns:
            bool: Success status
            
        What happens internally:
        1. Text is converted to embeddings (vectors)
        2. Both text and vectors are stored
        3. Metadata is attached for filtering
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Add the document to ChromaDB
            # ChromaDB automatically generates embeddings using our embedding function
            collection.add(
                documents=[content],  # The actual text
                ids=[document_id],    # Unique identifier
                metadatas=[metadata]  # Additional information for filtering
            )
            
            logger.info(f"✅ Document added: {document_id} to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add document {document_id}: {e}")
            return False
    
    def search_similar(
        self, 
        query: str, 
        collection_name: str = "transcripts",
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search text (can be in any supported language)
            collection_name: Which collection to search in
            n_results: How many results to return
            where_filter: Filter by metadata (e.g., language, date)
            
        Returns:
            List of similar documents with scores
            
        How semantic search works:
        1. Query is converted to embeddings (same as documents)
        2. ChromaDB calculates similarity between query and all documents
        3. Returns most similar documents ranked by similarity score
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter  # Optional filtering by metadata
            )
            
            # Format results for easier use
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                })
            
            logger.info(f"✅ Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str = "transcripts") -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Useful for:
        - Monitoring how much data we have
        - Understanding the content distribution
        - Debugging and optimization
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                return {"error": f"Collection '{collection_name}' not found"}
            
            count = collection.count()
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats for {collection_name}: {e}")
            return {"error": str(e)}
    
    def delete_document(self, document_id: str, collection_name: str = "transcripts") -> bool:
        """
        Delete a document from the collection.
        
        Useful for:
        - Removing outdated content
        - Correcting mistakes
        - Managing storage space
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection.delete(ids=[document_id])
            logger.info(f"✅ Document deleted: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {e}")
            return False
    
    def reset_collection(self, collection_name: str) -> bool:
        """
        Reset (clear all data from) a collection.
        
        ⚠️ WARNING: This deletes all data in the collection!
        Only use during development or when you want to start fresh.
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Delete the collection and recreate it
            self.client.delete_collection(collection_name)
            self._initialize_collections()
            
            logger.info(f"✅ Collection reset: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reset collection {collection_name}: {e}")
            return False


# Global instance for use across the application
# This ensures we have one database connection shared across all requests
vector_db = None


def get_vector_db() -> VectorDatabaseService:
    """
    Get the global vector database instance.
    
    This is a dependency function for FastAPI:
    - Creates the database connection once
    - Reuses it across all API requests
    - Ensures efficient resource usage
    """
    global vector_db
    if vector_db is None:
        vector_db = VectorDatabaseService()
    return vector_db


def initialize_vector_db() -> VectorDatabaseService:
    """
    Initialize the vector database service.
    
    Call this during application startup to:
    - Create database connections
    - Initialize collections
    - Verify everything is working
    """
    global vector_db
    vector_db = VectorDatabaseService()
    logger.info("🚀 Vector database service initialized")
    return vector_db
