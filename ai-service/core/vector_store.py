import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.eventos = []  # lista de { "id": X, "texto": "...", "embedding": [...] }

    def agregar_evento(self, event_id: str, texto: str, embedding: List[float]):
        self.eventos.append({
            "id": event_id,
            "texto": texto,
            "embedding": embedding
        })

    def similitud_coseno(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def buscar_similares(self, embedding_consulta: List[float], top_k: int = 3) -> List[Dict]:
        resultados = []

        for evento in self.eventos:
            score = self.similitud_coseno(embedding_consulta, evento["embedding"])
            resultados.append({
                "id": evento["id"],
                "texto": evento["texto"],
                "score": float(score)
            })

        resultados.sort(key=lambda x: x["score"], reverse=True)

        return resultados[:top_k]
