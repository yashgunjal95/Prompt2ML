# agents/chat_memory.py
import logging
from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from retriever.vector_store import VectorStore
from config.settings import GROQ_API_KEY, GROQ_MODEL
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class ChatMemory:
    """Handle conversational AI with memory and context retrieval."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
        self.vector_store = VectorStore()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = None
        self._initialize_chain()
    
    def _initialize_chain(self) -> None:
        """Initialize the conversational retrieval chain."""
        try:
            retriever = self.vector_store.get_retriever()
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("Chat chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chat chain: {e}")
            raise
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question and get an answer with context.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            if not self.qa_chain:
                self._initialize_chain()
            
            response = self.qa_chain.invoke({"question": question})
            
            logger.info(f"Question answered: {question[:50]}...")
            
            return {
                "answer": response.get("answer", "I'm sorry, I couldn't find an answer to your question."),
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "source_documents": []
            }
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

# Legacy function for backward compatibility
def get_chat_chain():
    """Legacy function to get chat chain."""
    chat_memory = ChatMemory()
    return chat_memory