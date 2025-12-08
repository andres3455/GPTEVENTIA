from huggingface_hub import InferenceClient
from config.settings import HF_API_KEY, DEFAULT_MODEL_TEXT

def get_llama_client():
        return InferenceClient(
            model=DEFAULT_MODEL_TEXT,
            token= HF_API_KEY
)