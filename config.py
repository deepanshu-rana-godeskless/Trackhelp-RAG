"""
Configuration settings for the RAG application.
"""
import os
from pathlib import Path


class Settings:
    """Application configuration settings."""
    
    # Directory paths
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    UPLOADS_DIR: str = "uploads"
    
    # Model configurations
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    
    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SIMILARITY_SEARCH_K: int = 3
    
    # CORS settings
    ALLOWED_ORIGINS: list = ["*"]  # Restrict in production
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: list = ["*"]
    ALLOWED_HEADERS: list = ["*"]
    
    def __init__(self):
        """Ensure required directories exist."""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()