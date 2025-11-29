# core/providers/jina_provider.py
import os
import requests

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

class JinaEmbeddingProvider:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or JINA_API_KEY
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY not set")

    def embed(self, text: str):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": "jina-embeddings-v3", "input": text}
        resp = requests.post(JINA_URL, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Jina returns data["data"][0]["embedding"]
        return data["data"][0]["embedding"]
