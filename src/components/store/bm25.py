from rank_bm25 import BM25Okapi
from typing import Dict, Any
from .base import BaseDatabase

class BM25Store(BaseDatabase):
    def __init__(self):
        super().__init__()
        self.indexes = {}  # {index_name: (BM25Okapi, [docs], [doc_ids], [metadatas])}

    def connect(self) -> None:
        self._is_connected = True

    def get_index(self, index_name: str) -> Any:
        return self.indexes.get(index_name, None)

    def create_index(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        text = document["text"]
        tokens = text.split()

        if index_name not in self.indexes:
            self.indexes[index_name] = ([], [], [])

        docs, doc_ids, metadatas = self.indexes[index_name]
        docs.append(tokens)
        doc_ids.append(doc_id)
        metadatas.append(document.get("metadata", {}))
        bm25 = BM25Okapi(docs)
        self.indexes[index_name] = (bm25, doc_ids, metadatas)
        return True

    def search(self, index_name: str, query: str, top_k: int = 5):
        index_data = self.indexes.get(index_name)
        if not index_data:
            return []

        bm25, doc_ids, metadatas = index_data
        query_tokens = query.split()
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(doc_ids, scores, metadatas),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for doc_id, score, meta in ranked[:top_k]:
            results.append({
                "id": doc_id,
                "score": float(score),
                "metadata": meta
            })
        return results

    def delete_index(self, index_name: str) -> None:
        if index_name in self.indexes:
            del self.indexes[index_name]

    def delete_document(self, index_name: str, doc_id: str) -> None:
        raise NotImplementedError("BM25 index does not support removing a single doc easily.")

    def update_document(self, index_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        raise NotImplementedError("BM25 index does not support updating in-place. Rebuild index.")

    def close(self) -> None:
        self._is_connected = False
