"""
Configuration management for Adhyatmik Intelligence AI Backend
Handles environment variables and application settings
"""

from typing import List, Optional
from pydantic import BaseSettings, Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    Uses Pydantic for validation and type safety.
    """
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        env="ALLOWED_ORIGINS"
    )
    
    # AI/ML Service API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_translate_api_key: Optional[str] = Field(default=None, env="GOOGLE_TRANSLATE_API_KEY")
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="adhyatmik-ai-index", env="PINECONE_INDEX_NAME")
    
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Security
    secret_key: str = Field(default="your_super_secret_key_here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Context7 Configuration
    context7_enabled: bool = Field(default=True, env="CONTEXT7_ENABLED")
    context7_max_tokens: int = Field(default=5000, env="CONTEXT7_MAX_TOKENS")
    
    # Embedding Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    
    # Translation Configuration
    default_language: str = Field(default="english", env="DEFAULT_LANGUAGE")
    supported_languages: List[str] = Field(
        default=["gujarati", "hindi", "english", "hinglish"],
        env="SUPPORTED_LANGUAGES"
    )
    
    # RAG Configuration
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    # Development Settings
    debug: bool = Field(default=True, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Parse environment variables as lists
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name in ['allowed_origins', 'supported_languages']:
                return [x.strip() for x in raw_val.split(',')]
            return cls.json_loads(raw_val)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency function to get application settings.
    Can be used with FastAPI's dependency injection.
    """
    return settings


# Utility functions for configuration validation
def validate_api_keys() -> dict:
    """
    Validate that required API keys are present.
    Returns a dictionary with validation results.
    """
    validation_results = {
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "google_translate": bool(settings.google_translate_api_key),
        "pinecone": bool(settings.pinecone_api_key and settings.pinecone_environment),
        "weaviate": bool(settings.weaviate_api_key) if settings.weaviate_url != "http://localhost:8080" else True
    }
    
    return validation_results


def get_vector_db_config() -> dict:
    """
    Get vector database configuration based on available settings.
    Returns the preferred vector DB configuration.
    """
    if settings.pinecone_api_key and settings.pinecone_environment:
        return {
            "type": "pinecone",
            "api_key": settings.pinecone_api_key,
            "environment": settings.pinecone_environment,
            "index_name": settings.pinecone_index_name
        }
    elif settings.weaviate_api_key:
        return {
            "type": "weaviate",
            "url": settings.weaviate_url,
            "api_key": settings.weaviate_api_key
        }
    else:
        return {
            "type": "chroma",
            "persist_directory": settings.chroma_persist_directory
        }


def create_directories():
    """
    Create necessary directories for the application.
    """
    directories = [
        settings.chroma_persist_directory,
        "logs",
        "data/uploads",
        "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Initialize directories on import
create_directories()

