"""
Embedding Service for Multilingual Text Processing
================================================

This service handles converting text into numerical vectors (embeddings).

Key Learning Points:
1. **What are embeddings?** Mathematical representations of text that capture meaning
2. **Why multilingual?** Our model understands Gujarati, Hindi, English, and more
3. **How do they work?** Similar meanings = similar numbers = findable content

Example:
- "હેલો" (Gujarati) and "Hello" (English) will have similar embeddings
- This allows cross-language search and understanding
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text embedding generation for multilingual content.
    
    This service:
    - Converts text to numerical vectors
    - Supports multiple languages including Gujarati
    - Caches embeddings for better performance
    - Provides similarity calculations
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: The sentence transformer model to use
            
        Why this model?
        - **paraphrase-multilingual**: Trained on 50+ languages including Gujarati
        - **mpnet**: Advanced architecture that understands context well
        - **base-v2**: Optimized version with good performance/accuracy balance
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = Path("./embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"✅ Embedding service initialized with model: {model_name}")
    
    def _load_model(self):
        """
        Load the sentence transformer model.
        
        What happens here:
        1. Downloads the model if not already present (first time only)
        2. Loads it into memory for fast inference
        3. Sets up GPU usage if available (faster processing)
        """
        try:
            logger.info(f"🔄 Loading embedding model: {self.model_name}")
            
            # Load the model
            self.model = SentenceTransformer(self.model_name)
            
            # Use GPU if available for faster processing
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            logger.info(f"✅ Model loaded successfully on device: {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text in any supported language
            use_cache: Whether to cache the result for faster future access
            
        Returns:
            numpy array representing the text as a vector
            
        How it works:
        1. Text is tokenized (broken into pieces the model understands)
        2. Model processes tokens through neural networks
        3. Output is a 768-dimensional vector capturing meaning
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(768)  # Return zero vector for empty text
        
        # Create cache key based on text content
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache first (faster)
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"📁 Loaded embedding from cache for text: {text[:50]}...")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        try:
            # Generate new embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache for future use
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)
                except Exception as e:
                    logger.warning(f"Failed to cache embedding: {e}")
            
            logger.debug(f"✅ Generated embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return np.zeros(768)  # Return zero vector on error
    
    def generate_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            
        Returns:
            List of embeddings
            
        Why batch processing?
        - More efficient than processing one by one
        - Better GPU utilization
        - Faster overall processing for large datasets
        """
        if not texts:
            return []
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append(np.zeros(768))
                continue
                
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if use_cache and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    embeddings.append(embedding)
                except Exception:
                    # If cache fails, add to uncached list
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batch
        if uncached_texts:
            try:
                logger.info(f"🔄 Generating embeddings for {len(uncached_texts)} texts")
                batch_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                
                # Store results and cache them
                for i, embedding in enumerate(batch_embeddings):
                    idx = uncached_indices[i]
                    embeddings[idx] = embedding
                    
                    # Cache the embedding
                    if use_cache:
                        try:
                            text = uncached_texts[i]
                            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
                            cache_file = self.cache_dir / f"{cache_key}.pkl"
                            with open(cache_file, 'wb') as f:
                                pickle.dump(embedding, f)
                        except Exception as e:
                            logger.warning(f"Failed to cache embedding: {e}")
                
                logger.info(f"✅ Generated {len(batch_embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"❌ Batch embedding generation failed: {e}")
                # Fill remaining with zero vectors
                for idx in uncached_indices:
                    if embeddings[idx] is None:
                        embeddings[idx] = np.zeros(768)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1 (1 = identical meaning)
            
        What is cosine similarity?
        - Measures the angle between two vectors
        - 0 = completely different meanings
        - 1 = identical meanings
        - Works well for text embeddings
        """
        try:
            # Normalize vectors to unit length
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query.
        
        Args:
            query_embedding: The query vector
            candidate_embeddings: List of candidate vectors
            top_k: How many top results to return
            
        Returns:
            List of similarity results with scores and indices
        """
        if not candidate_embeddings:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Useful for:
        - Debugging and monitoring
        - Understanding model capabilities
        - API documentation
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": 768,
            "max_sequence_length": 512,
            "supported_languages": [
                "english", "hindi", "gujarati", "bengali", "tamil", 
                "telugu", "marathi", "urdu", "spanish", "french", 
                "german", "chinese", "japanese", "korean", "arabic"
            ],
            "model_type": "sentence-transformer",
            "architecture": "mpnet",
            "cache_enabled": True,
            "cache_directory": str(self.cache_dir)
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the embedding cache.
        
        Use when:
        - Cache becomes too large
        - Model is updated
        - Testing different configurations
        """
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            
            logger.info("✅ Embedding cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False


# Global instance
embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    This ensures we have one model loaded in memory,
    shared across all API requests for efficiency.
    """
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service


def initialize_embedding_service() -> EmbeddingService:
    """
    Initialize the embedding service during application startup.
    """
    global embedding_service
    embedding_service = EmbeddingService()
    logger.info("🚀 Embedding service initialized")
    return embedding_service
