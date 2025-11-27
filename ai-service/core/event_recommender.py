from core.embedding import generar_embedding
from core.vector_store import VectorStore

class EventRecommender:
    def __init__(self):
        self.store = VectorStore()

    def cargar_eventos(self, eventos: list):
        """
        Carga eventos desde una lista de diccionarios.
        Cada evento debe tener id y descripcion.
        """
        for ev in eventos:
            emb = generar_embedding(ev["descripcion"])
            self.store.agregar_evento(ev["id"], ev["descripcion"], emb)

    def recomendar_eventos(self, prompt_usuario: str, top_k: int = 3):
        """
        Genera recomendaciones a partir del prompt del usuario.
        """
        embedding_usuario = generar_embedding(prompt_usuario)
        similares = self.store.buscar_similares(embedding_usuario, top_k=top_k)
        return similares
