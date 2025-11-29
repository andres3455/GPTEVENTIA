from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

print("[*] Cargando modelo Qwen 3B...")

tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def generate_text(prompt: str, max_tokens: int = 200):
    """Genera texto estilo chat usando Qwen."""
    try:
        # Formato especial de conversación para Qwen
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(output[0], skip_special_tokens=True)

    except Exception as e:
        print(f"[ERROR] generate_text(): {e}")
        return None
