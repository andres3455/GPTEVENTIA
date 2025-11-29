# core/reranker.py
from typing import List, Dict
from core.hf_client import hf_generate
import json
import re

MODEL_RERANK = "mistralai/Mistral-7B-Instruct-v0.2"  # recommended free model

SYSTEM_PROMPT = (
    "You are a ranking assistant. Given a user query and a short list of candidate events, "
    "score each candidate for relevance in range [0.0,1.0] and provide a one-sentence explanation. "
    "Return a JSON array of objects: [{\"id\": \"...\", \"score\": 0.87, \"explanation\":\"...\"}, ...] and nothing else."
)

def rerank_with_hf(user_query: str, candidates: List[Dict]) -> List[Dict]:
    # prepare candidates compact
    compact = []
    for c in candidates:
        compact.append({
            "id": c["id"],
            "title": c.get("titulo","")[:200],
            "description": c.get("descripcion","")[:400]
        })
    prompt_user = f"User query: {user_query}\nCandidates: {compact}"
    prompt = SYSTEM_PROMPT + "\n\n" + prompt_user
    raw = hf_generate(MODEL_RERANK, prompt, max_tokens=300)
    # HF may return text — try to parse JSON
    out = raw.strip()
    try:
        arr = json.loads(out)
    except Exception:
        m = re.search(r"(\[.*\])", out, re.S)
        if not m:
            raise RuntimeError("Reranker returned non-JSON: " + out)
        arr = json.loads(m.group(1))
    # normalize entries
    normalized = []
    for a in arr:
        normalized.append({
            "id": a.get("id"),
            "score": float(a.get("score", 0.0)),
            "explanation": a.get("explanation", "")[:240]
        })
    normalized.sort(key=lambda x: x["score"], reverse=True)
    return normalized
