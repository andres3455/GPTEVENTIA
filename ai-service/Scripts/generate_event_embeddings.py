# scripts/generate_event_embeddings.py
import json
import os
import sys

# Ensure project root in sys.path when running script from scripts/
CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from core.event_recommender import EventRecommender

def main():
    data_path = os.path.join(ROOT, "data", "eventos.json")

    # cargar eventos
    with open(data_path, "r", encoding="utf-8") as f:
        eventos = json.load(f)

    reco = EventRecommender()

    print("[*] Indexando eventos (generando embeddings)...")
    reco.embed_events(eventos)

    print("[*] Listo. Ejecutando prueba de recomendación...\n")

    prompt = "Quiero una actividad tranquila para relajarme después del trabajo"

    resultados = reco.recommend(eventos, prompt, top_k=3)

    print("=== RECOMENDACIONES ===")
    for r in resultados:
        e = r["event"]
        print(f"- {e['titulo']}  (score: {r['score']:.4f})")


    # -------------------------------------------------------
    # Generación de 1 evento nuevo
    # -------------------------------------------------------
    print("\n=== GENERANDO UN NUEVO EVENTO ===")

    try:
        nuevo_evento = reco.generate_and_index("Quiero una experiencia relajante en la naturaleza")

        if isinstance(nuevo_evento, list):
            nuevo_evento = nuevo_evento[0]

        print("Evento generado:")
        print(f"  Título: {nuevo_evento.get('titulo')}")
        print(f"  Descripción: {nuevo_evento.get('descripcion')}")
        print(f"  Categoría: {nuevo_evento.get('categoria')}")
    except Exception as e:
        print("[!] No se pudo generar el evento:", e)


if __name__ == "__main__":
    main()
