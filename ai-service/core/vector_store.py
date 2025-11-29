# core/vector_store.py
import numpy as np
from typing import List, Dict

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

class VectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.metadatas: List[Dict] = []

    def add(self, vector: List[float], metadata: Dict):
        self.vectors.append(np.array(vector, dtype=float))
        self.metadatas.append(metadata)

    def top_k(self, query_vector: List[float], k: int = 5):
        q = np.array(query_vector, dtype=float)
        scores = []
        for v, m in zip(self.vectors, self.metadatas):
            s = cosine_similarity(q, v)
            scores.append({"score": s, "metadata": m})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:k]

    def all(self):
        return [{"score": None, "metadata": m} for m in self.metadatas]
