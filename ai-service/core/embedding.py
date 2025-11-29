# core/embedding.py
"""
Interfaz de embeddings. Cambia el provider aquí para usar Jina o HuggingFace.
"""

from core.embedding_providers.jina_provider import JinaEmbeddingProvider
from core.embedding_providers.huggingface_provider import HuggingFaceEmbeddingProvider
import os

# Selección de provider por variable de entorno: JINA or HF
DEFAULT_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "JINA").upper()

if DEFAULT_PROVIDER == "HF" or DEFAULT_PROVIDER == "HUGGINGFACE":
    provider = HuggingFaceEmbeddingProvider()
else:
    provider = JinaEmbeddingProvider()

def generar_embedding(texto: str):
    # normalize text a bit
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.replace("\n", " ").strip()
    return provider.embed(texto)
