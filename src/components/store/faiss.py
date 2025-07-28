# src/store/faiss_store.py

import faiss
import numpy as np
from typing import Dict, Any
from .base import BaseDatabase

class FAISS(BaseDatabase):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.indexes = {}  # {index_name: (faiss.IndexFlatIP, [doc_ids], [metadatas])}

    def connect(self) -> None:
        self._is_connected = True

    def get_index(self, index_name: str) -> Any:
        return self.indexes.get(index_name, None)

    def create_index(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        embedding = np.array(document["embedding"]).astype('float32').reshape(1, -1)

        if index_name not in self.indexes:
            index = faiss.IndexFlatIP(self.dim)
            self.indexes[index_name] = (index, [], [])

        index, doc_ids, metadatas = self.indexes[index_name]
        index.add(embedding)
        doc_ids.append(doc_id)
        metadatas.append(document.get("metadata", {}))
        return True

    def search(self, index_name: str, query_embedding: list, top_k: int = 5):
        index_data = self.indexes.get(index_name)
        if not index_data:
            return []
        
        index, doc_ids, metadatas = index_data
        query = np.array(query_embedding).astype('float32').reshape(1, -1)
        scores, I = index.search(query, top_k)
        results = []
        for idx, score in zip(I[0], scores[0]):
            if idx < len(doc_ids):
                results.append({
                    "id": doc_ids[idx],
                    "score": float(score),
                    "metadata": metadatas[idx]
                })
        return results

    def delete_index(self, index_name: str) -> None:
        if index_name in self.indexes:
            del self.indexes[index_name]

    def delete_document(self, index_name: str, doc_id: str) -> None:
        raise NotImplementedError("FAISS index does not support deleting a single vector easily.")

    def update_document(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        raise NotImplementedError("FAISS index does not support updating in-place. Delete & re-add.")

    def close(self) -> None:
        self._is_connected = False
