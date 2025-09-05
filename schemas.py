"""
Pydantic schemas for request/response validation.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    """Request schema for AI prompt queries."""
    prompt: str = Field(..., min_length=1, description="User's question or prompt")


class UploadResponse(BaseModel):
    """Response schema for file upload."""
    message: str


class FileListResponse(BaseModel):
    """Response schema for listing files."""
    files: List[str]


class ChunkDetail(BaseModel):
    """Individual document chunk with metadata."""
    chunk: str
    metadata: Dict[str, Any]


class FileDetailsResponse(BaseModel):
    """Response schema for file details."""
    file: str
    chunks: List[ChunkDetail]


class DeleteResponse(BaseModel):
    """Response schema for file deletion."""
    message: str


class AskResponse(BaseModel):
    """Response schema for AI queries."""
    answer: str


class InsightsResponse(BaseModel):
    """Response schema for insights generation."""
    answer: str