# event_recommender.py
import numpy as np
from core.hf_client import embed_text
from core.generator import EventGenerator  # <-- necesario para generar eventos IA

class EventRecommender:

    def embed_events(self, events):
        embeddings = []

        for ev in events:
            print(ev)
            vector = embed_text(ev["descripcion"])

            if vector is None:
                embeddings.append(np.zeros(384))
                continue

            vec = np.array(vector).flatten()
            target_dim = 384 if len(vec) >= 384 else len(vec)
            vec = vec[:target_dim]

            embeddings.append(vec)

        return np.vstack(embeddings)


    def embed_user_preferences(self, preferences: str):
        vector = embed_text(preferences)
        if vector is None:
            return np.zeros(384)

        vec = np.array(vector).flatten()
        target_dim = 384 if len(vec) >= 384 else len(vec)
        return vec[:target_dim]


    def recommend(self, events, user_preferences, top_k=3):
        """Recomienda eventos usando coseno + embeddings."""
        event_vectors = self.embed_events(events)
        user_vec = self.embed_user_preferences(user_preferences)

        scores = event_vectors @ user_vec.T
        scores = scores / (
            np.linalg.norm(event_vectors, axis=1) * np.linalg.norm(user_vec) + 1e-8
        )

        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "event": events[i],
                "score": float(scores[i])
            }
            for i in idx
        ]


# ---------------------------------------------------------
# 🚀 NUEVO: generate_and_index()
# ---------------------------------------------------------
    def generate_and_index(self, prompt: str, max_events: int = 1):
        """
        Genera nuevos eventos con EventGenerator, los indexa y devuelve la lista.
        """
        generator = EventGenerator()

        # generar varios eventos usando el generator
        new_events = generator.generate_events(prompt, num_events=max_events)

        if not hasattr(self, "events"):
            self.events = []

        if not hasattr(self, "event_vectors"):
            self.event_vectors = None

        indexed = []

        for ev in new_events:
            descripcion = ev.get("descripcion") or ev.get("description")
            if not descripcion:
                descripcion = "Sin descripción"

            vec = embed_text(descripcion)
            if vec is None:
                vec = np.zeros(384)

            vec = np.array(vec).flatten()
            if len(vec) > 384:
                vec = vec[:384]

            # agregar al índice
            self.events.append(ev)

            if self.event_vectors is None:
                self.event_vectors = np.array([vec])
            else:
                self.event_vectors = np.vstack([self.event_vectors, vec])

            indexed.append(ev)

        return indexed