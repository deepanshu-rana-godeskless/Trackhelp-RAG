"""
Document processing service for handling PDF files and text chunking.
"""
import os
import shutil
import uuid
from typing import List
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings


class DocumentProcessorService:
    """Service for processing and managing uploaded documents."""
    
    def __init__(self):
        """Initialize the document processor with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to disk with unique filename.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Path to the saved file
        """
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = Path(settings.UPLOADS_DIR) / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return str(file_path)
    
    def process_pdf(self, file_path: str, source_name: str) -> List[Document]:
        """
        Process PDF file into document chunks.
        
        Args:
            file_path: Path to the PDF file
            source_name: Name to use as source identifier
            
        Returns:
            List of processed document chunks
        """
        # Load PDF documents
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add source metadata to each chunk
        for chunk in chunks:
            chunk.metadata["src"] = source_name
        
        return chunks
    
    def cleanup_file(self, file_path: str) -> None:
        """
        Remove temporary file from disk.
        
        Args:
            file_path: Path to the file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            # Log error in production
            pass
    
    async def process_uploaded_pdf(self, file: UploadFile) -> List[Document]:
        """
        Complete pipeline for processing uploaded PDF file.
        
        Args:
            file: Uploaded PDF file
            
        Returns:
            List of processed document chunks
            
        Raises:
            ValueError: If file is not a PDF
        """
        if not file.filename.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")
        
        # Save uploaded file
        file_path = await self.save_uploaded_file(file)
        
        try:
            # Process PDF into chunks
            chunks = self.process_pdf(file_path, file.filename)
            return chunks
        finally:
            # Clean up temporary file
            self.cleanup_file(file_path)