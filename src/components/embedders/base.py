from abc import abstractmethod, ABC
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import torch
import numpy as np

class BaseEmbedder(ABC):
    """Base class for all embedders"""
    @abstractmethod
    def embed_query(self, text: Union[str, List[str]]) -> np.ndarray:
        """Embed text into vector space"""
        pass           
    @abstractmethod
    def embed_image(self, image_path: Union[str, List[str]]) -> np.ndarray:
        """Embed image into vector space"""
        pass

    @abstractmethod 
    def embed_multimodal(self, text: str, image_path: str) -> np.ndarray:
        """Embed both text and image into joint vector space"""
        pass