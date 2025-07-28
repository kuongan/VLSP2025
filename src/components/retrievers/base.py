from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalMode(Enum):
    TEXT_ONLY = "text_only" 
    MULTIMODAL = "multimodal"  
    ENSEMBLE = "ensemble"  
class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations with multiple retrieval strategies.
    """

    def __init__(self, 
                 text_store: Any = None,
                 image_store: Any = None,
                 multimodal_store: Any = None):
        """
        Initialize the retriever with different types of stores.
        :param text_store: Store for text-based retrieval
        :param image_store: Store for image-based retrieval
        :param multimodal_store: Store for multimodal retrieval
        """
        self.text_store = text_store
        self.image_store = image_store
        self.multimodal_store = multimodal_store
        self.mode = RetrievalMode.MULTIMODAL  
    @abstractmethod
    def ground_to_text(self, image: Union[str, Image.Image]) -> str:
        """
        Convert image content to textual description using grounding models.
        :param image: Input image
        :return: Textual description of the image
        """
        pass

    @abstractmethod
    def retrieve_from_text_store(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from text-only store
        """
        pass

    @abstractmethod
    def retrieve_from_image_store(
        self,
        image: Union[str, Image.Image],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from image-only store
        """
        pass

    @abstractmethod
    def retrieve_from_multimodal_store(
        self,
        query: str,
        image: Optional[Union[str, Image.Image]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from multimodal store
        """
        pass

    def ensemble_results(
        self,
        results_list: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine results from multiple retrievers
        """
        # Default implementation using simple score averaging
        # Can be overridden for more sophisticated fusion
        if not weights:
            weights = [1.0] * len(results_list)
            
        # Normalize weights
        weights = [w/sum(weights) for w in weights]
        
        # Combine scores
        combined_scores = {}
        for results, weight in zip(results_list, weights):
            for doc in results:
                doc_id = doc['id']
                score = doc.get('score', 0.0) * weight
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {'doc': doc, 'score': 0.0}
                combined_scores[doc_id]['score'] += score
                
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        return [item['doc'] for item in sorted_results]
    def retrieve(
        self,
        query: str,
        image: Optional[Union[str, Image.Image]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieve method that handles different modes
        """
        if self.mode == RetrievalMode.TEXT_ONLY:
            # Convert everything to text
            text_query = query
            if image is not None:
                img_desc = self.ground_to_text(image)
                text_query = f"{text_query} {img_desc}"
            return self.retrieve_from_text_store(text_query, top_k)
            
        elif self.mode == RetrievalMode.MULTIMODAL:
            return self.retrieve_from_multimodal_store(query, image, top_k)
            
        elif self.mode == RetrievalMode.ENSEMBLE:
            # Get results from all stores
            text_results = self.retrieve_from_text_store(query, top_k)
            image_results = self.retrieve_from_image_store(image, top_k) if image else []
            multimodal_results = self.retrieve_from_multimodal_store(query, image, top_k)
            
            # Combine results
            return self.ensemble_results(
                [text_results, image_results, multimodal_results],
                weights=[0.3, 0.3, 0.4]
            )            
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")
    @abstractmethod
    def rerank(
        self,
        old_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """ rerank """
        pass
