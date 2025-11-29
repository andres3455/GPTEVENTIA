# core/providers/huggingface_provider.py
import os
import requests

HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_INFERENCE_URL_BASE = "https://api-inference.huggingface.co/models"

class HuggingFaceEmbeddingProvider:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or HF_API_KEY
        self.model = model or HF_EMBEDDING_MODEL
        if not self.api_key:
            raise RuntimeError("HF_API_KEY not set")

    def embed(self, text: str):
        url = f"{HF_INFERENCE_URL_BASE}/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Depending on model + HF response shape: often returns list of embeddings or nested structure.
        # Many HF embedding endpoints return a list of floats under data[0] or directly first item.
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace error: {data['error']}")
        # Typical response: list of floats or list of lists
        if isinstance(data, list):
            # Some models return [[...]] or [[...], ...]
            first = data[0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
        # Fallback: try to extract embed field
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        raise RuntimeError("Unexpected HF embedding response shape: " + str(data))
