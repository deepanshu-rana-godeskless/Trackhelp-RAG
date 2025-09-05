"""
API routes for the RAG application.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from schemas import (
    PromptRequest, UploadResponse, FileListResponse, 
    FileDetailsResponse, DeleteResponse, AskResponse, InsightsResponse
)
from services.vector_store import VectorStoreService
from services.llm import LLMService
from services.document_processor import DocumentProcessorService

# Initialize services
vector_store_service = VectorStoreService()
llm_service = LLMService()
document_processor = DocumentProcessorService()

# Create router
router = APIRouter()


@router.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a PDF file for RAG.
    
    Args:
        file: PDF file to upload and process
        
    Returns:
        Success message with filename
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Process uploaded PDF
        chunks = await document_processor.process_uploaded_pdf(file)
        
        # Add to vector store
        vector_store_service.add_documents(chunks)
        
        return UploadResponse(message=f"Uploaded and processed {file.filename}")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/files/", response_model=FileListResponse)
async def list_files():
    """
    List all processed files in the vector store.
    
    Returns:
        List of source file names
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        files = vector_store_service.get_source_files()
        return FileListResponse(files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve files: {str(e)}")


@router.get("/files/{file_name}/details", response_model=FileDetailsResponse)
async def file_details(file_name: str):
    """
    Get detailed information about a specific file's chunks.
    
    Args:
        file_name: Name of the file to get details for
        
    Returns:
        File details with all chunks and metadata
        
    Raises:
        HTTPException: If file not found or retrieval fails
    """
    try:
        chunks = vector_store_service.get_documents_by_source(file_name)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileDetailsResponse(file=file_name, chunks=chunks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file details: {str(e)}")


@router.delete("/files/{file_name}", response_model=DeleteResponse)
async def delete_file(file_name: str):
    """
    Delete all vectors associated with a specific file.
    
    Args:
        file_name: Name of the file to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If file not found or deletion fails
    """
    try:
        # Get IDs to delete
        ids_to_delete = vector_store_service.get_ids_by_source(file_name)
        
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from vector store
        vector_store_service.delete_by_ids(ids_to_delete)
        
        return DeleteResponse(message=f"Deleted all vectors for {file_name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.post("/ask/", response_model=AskResponse)
async def ask(req: PromptRequest):
    """
    Ask a question using RAG (Retrieval-Augmented Generation).
    
    Args:
        req: Request containing the user's question
        
    Returns:
        AI-generated answer based on relevant documents
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Retrieve relevant documents
        relevant_docs = vector_store_service.similarity_search(req.prompt)
        
        # Generate context-aware response
        answer = llm_service.generate_context_aware_response(req.prompt, relevant_docs)
        
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@router.post("/insights/", response_model=InsightsResponse)
async def insights(req: PromptRequest):
    """
    Generate insights or general responses without document context.
    
    Args:
        req: Request containing the prompt for insights
        
    Returns:
        AI-generated response
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        answer = llm_service.generate_response(req.prompt)
        return InsightsResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.get("/debug/ollama")
async def debug_ollama():
    """Debug endpoint to test Ollama LLM connectivity."""
    try:
        response = llm_service.generate_response("Hello, respond with just 'OK'")
        return {"status": "success", "llm_response": response}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/debug/embeddings")
async def debug_embeddings():
    """Debug endpoint to test embeddings connectivity."""
    try:
        # Test embeddings by trying to search (even if no docs exist)
        test_docs = vector_store_service.similarity_search("test", k=1)
        return {"status": "success", "docs_count": len(test_docs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/debug/status")
async def debug_status():
    """Debug endpoint to check overall system status."""
    status = {
        "vector_store": "unknown",
        "llm": "unknown",
        "embeddings": "unknown"
    }
    
    # Test vector store
    try:
        vector_store_service.get_source_files()
        status["vector_store"] = "working"
    except Exception as e:
        status["vector_store"] = f"error: {str(e)}"
    
    # Test LLM
    try:
        response = llm_service.generate_response("Hi")
        status["llm"] = "working"
    except Exception as e:
        status["llm"] = f"error: {str(e)}"
    
    # Test embeddings
    try:
        vector_store_service.similarity_search("test", k=1)
        status["embeddings"] = "working"
    except Exception as e:
        status["embeddings"] = f"error: {str(e)}"
    
    return status