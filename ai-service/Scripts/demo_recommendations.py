# scripts/demo_recommendations.py
import json, os, sys
CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from core.event_recommender import EventRecommender

def run_demo():
    with open(os.path.join(ROOT, "data", "eventos.json"), "r", encoding="utf-8") as f:
        eventos = json.load(f)
    reco = EventRecommender()
    reco.load_events(eventos)

    prompt = input("Describe qué quieres hacer: ")
    results = reco.recommend(prompt, top_k=5)
    print("\nResultados:")
    for r in results:
        print(f"- {r.get('titulo')}  (explanation: {r.get('explanation','')})")

if __name__ == "__main__":
    run_demo()
