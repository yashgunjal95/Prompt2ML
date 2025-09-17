# retriever/vector_store.py
import logging
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DIR

logger = logging.getLogger(__name__)

class VectorStore:
    """Handle vector database operations for document storage and retrieval."""
    
    def __init__(self, collection_name: str = "prompt2ml_store"):
        self.collection_name = collection_name
        self.embeddings = None
        self.db = None
        self._initialize_embeddings()
        self._initialize_database()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the embeddings model."""
        try:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            # Fallback to a simpler embedding if Ollama fails
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                logger.info("Fallback embeddings initialized")
            except Exception as fallback_error:
                logger.error(f"Fallback embeddings also failed: {fallback_error}")
                raise
    
    def _initialize_database(self) -> None:
        """Initialize the vector database."""
        try:
            self.db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR)
            )
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            self.db.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def store_context(self, prompt: str, dataset_summary: str) -> None:
        """
        Store prompt and dataset context in vector store.
        
        Args:
            prompt: User's original prompt
            dataset_summary: Summary of the dataset
        """
        try:
            documents = [
                Document(
                    page_content=prompt,
                    metadata={"type": "user_prompt", "source": "user_input"}
                ),
                Document(
                    page_content=dataset_summary,
                    metadata={"type": "dataset_summary", "source": "data_analysis"}
                )
            ]
            
            self.add_documents(documents)
            logger.info("Context stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            Retriever object for the vector store
        """
        try:
            if search_kwargs is None:
                search_kwargs = {"k": 5}
            
            retriever = self.db.as_retriever(search_kwargs=search_kwargs)
            logger.info("Retriever created successfully")
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.db.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # This is a workaround as Chroma doesn't have a direct clear method
            self.db.delete_collection()
            self._initialize_database()
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

# Legacy functions for backward compatibility
def get_vector_store():
    """Legacy function to get vector store."""
    return VectorStore().db

def store_context(prompt: str, df_summary: str):
    """Legacy function to store context."""
    vector_store = VectorStore()
    vector_store.store_context(prompt, df_summary)

def get_retriever():
    """Legacy function to get retriever."""
    vector_store = VectorStore()
    return vector_store.get_retriever()