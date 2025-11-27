# test_reco.py

from core.embedding import EmbeddingEngine
from core.vector_store import VectorStore
from core.event_recommender import EventRecommender
import json

# ----- Inicializar motor -----
embedding = EmbeddingEngine(api_key="TU_API_KEY")
store = VectorStore()

# Cargar eventos y embeddings generados
with open("data/eventos.json", "r") as f:
    eventos = json.load(f)

# OJO: este ejemplo asume que ya generaste y guardaste embeddings
# Aquí generamos los embeddings en caliente, solo para prueba
for e in eventos:
    text = f"{e['name']} {e['description']} {' '.join(e['tags'])}"
    vec = embedding.get_embedding(text)
    store.add_item(e["id"], e["name"], vec)

recommender = EventRecommender(embedding, store)

# ----- Probar -----
prompt = "quiero una actividad relajante para liberar estrés"
resultados = recommender.recommend(prompt, top_k=3)

print(resultados)
