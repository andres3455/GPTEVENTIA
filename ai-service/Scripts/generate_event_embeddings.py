import json
import sys
import os

# --- FIX IMPORTS (primero SIEMPRE) ---
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FOLDER)
sys.path.append(PROJECT_ROOT)
# --------------------------------------

# Ahora sí puedes importar core
from core.event_recommender import EventRecommender


def main():
    # Cargar eventos base
    with open("./data/Eventos.json", "r", encoding="utf-8") as f:
        eventos = json.load(f)

    recomendador = EventRecommender()
    recomendador.cargar_eventos(eventos)

    prompt = "Quiero una actividad tranquila para relajarme"
    recomendaciones = recomendador.recomendar_eventos(prompt)

    print("\nRecomendaciones:")
    for r in recomendaciones:
        print(f"- {r['texto']} (score: {r['score']:.4f})")


if __name__ == "__main__":
    main()
