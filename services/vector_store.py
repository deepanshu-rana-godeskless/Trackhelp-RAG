"""
Vector store service for managing document embeddings and retrieval.
"""
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

from config import settings


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self):
        """Initialize vector store with embeddings."""
        self.embedding_function = OllamaEmbeddings(model=settings.EMBED_MODEL)
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of langchain Document objects to add
        """
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        k = k or settings.SIMILARITY_SEARCH_K
        return self.vector_store.similarity_search(query, k=k)
    
    def get_all_documents(self, include: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve all documents from the vector store.
        
        Args:
            include: List of fields to include in the response
            
        Returns:
            Dictionary containing documents and metadata
        """
        include = include or []
        return self.vector_store.get(include=include)
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self.vector_store.delete(ids=ids)
        self.vector_store.persist()
    
    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Get all documents from a specific source file.
        
        Args:
            source_name: Name of the source file
            
        Returns:
            List of documents with their metadata
        """
        results = self.get_all_documents(include=["metadatas", "documents"])
        documents = []
        
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            if meta.get("src") == source_name:
                documents.append({"chunk": doc, "metadata": meta})
        
        return documents
    
    def get_source_files(self) -> List[str]:
        """
        Get list of all unique source files in the vector store.
        
        Returns:
            List of source file names
        """
        results = self.get_all_documents(include=["metadatas"])
        sources = set()
        
        for metadata_list in results.get("metadatas", []):
            if isinstance(metadata_list, dict) and "src" in metadata_list:
                sources.add(metadata_list["src"])
            elif isinstance(metadata_list, list):
                for meta in metadata_list:
                    if isinstance(meta, dict) and "src" in meta:
                        sources.add(meta["src"])
        
        return list(sources)
    
    def get_ids_by_source(self, source_name: str) -> List[str]:
        """
        Get all document IDs for a specific source file.
        
        Args:
            source_name: Name of the source file
            
        Returns:
            List of document IDs
        """
        results = self.get_all_documents(include=["metadatas"])
        ids_to_return = []
        
        for doc_id, meta in zip(results.get("ids", []), results.get("metadatas", [])):
            if meta.get("src") == source_name:
                ids_to_return.append(doc_id)
        
        return ids_to_return