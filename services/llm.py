"""
Large Language Model service for generating responses.
"""
from typing import List
from langchain_ollama import OllamaLLM
from langchain.schema import Document

from config import settings


class LLMService:
    """Service for interacting with the Large Language Model."""
    
    def __init__(self):
        """Initialize the LLM with configured model."""
        self.llm = OllamaLLM(model=settings.LLM_MODEL)
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Generated response string
        """
        return self.llm.invoke(prompt)
    
    def generate_context_aware_response(self, question: str, context_documents: List[Document]) -> str:
        """
        Generate a response using retrieved context documents.
        
        Args:
            question: User's question
            context_documents: List of relevant documents for context
            
        Returns:
            Generated response string
        """
        if not context_documents:
            return self.generate_response(question)
        
        # Combine document content as context
        context = "\n\n".join([doc.page_content for doc in context_documents])
        
        # Create context-aware prompt
        prompt = self._create_rag_prompt(context, question)
        
        return self.generate_response(prompt)
    
    def _create_rag_prompt(self, context: str, question: str) -> str:
        """
        Create a RAG (Retrieval-Augmented Generation) prompt.
        
        Args:
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        return f"""Answer the question using the context below. If the answer cannot be found in the context, please say so clearly.

Context:
{context}

Question: {question}

Answer:"""