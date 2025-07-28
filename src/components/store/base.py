from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseDatabase(ABC):
    """Abstract base class for all database implementations"""
    
    def __init__(self):
        self._connection = None
        self._is_connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Set up database connection"""
        pass
    
    @abstractmethod
    def get_index(self, index_name: str) -> Any:
        """Get index by name"""
        pass
    
    @abstractmethod
    def create_index(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        """Add document to collection"""
        pass
    
    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """Delete index"""
        pass
    
    @abstractmethod
    def delete_document(self, index_name: str, doc_id: str) -> None:
        """Delete document by ID"""
        pass
    
    @abstractmethod
    def update_document(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        """Update document by ID"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Disconnect from the database"""
        pass
    