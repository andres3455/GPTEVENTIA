import os
import requests

JINA_API_KEY = os.getenv("JINA_API_KEY")

if not JINA_API_KEY:
    raise Exception("❌ No existe la variable de entorno JINA_API_KEY")

JINA_URL = "https://api.jina.ai/v1/embeddings"

def generar_embedding(texto: str):
    """
    Genera un embedding usando Jina Embeddings v3 (gratis)
    """

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "jina-embeddings-v3",
        "input": texto
    }

    response = requests.post(JINA_URL, json=body, headers=headers)

    if response.status_code != 200:
        print("Error Jina:", response.text)
        raise Exception("❌ No se pudo generar el embedding con Jina")

    data = response.json()
    return data["data"][0]["embedding"]
