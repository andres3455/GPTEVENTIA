# event_generator.py
from core.hf_client import generate_text
import json

class EventGenerator:

    def generate_event(self, category: str):
        """
        Genera un solo evento basado en categoría.
        """
        prompt = f"""
        Crea un evento realista basado en la categoría: {category}.
        Devuélvelo en formato JSON EXACTO, así:
        {{
            "titulo": "...",
            "descripcion": "...",
            "categoria": "{category}"
        }}
        NO agregues explicaciones ni texto adicional.
        """

        response = generate_text(prompt, max_tokens=300)

        if not response:
            return {
                "titulo": "Evento no disponible",
                "descripcion": "No se pudo generar un evento",
                "categoria": category
            }

        # Intento fuerte: parsear JSON correctamente
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_text = response[start:end]
            return json.loads(json_text)
        except:
            # fallback seguro
            return {
                "titulo": f"Evento generado: {category}",
                "descripcion": response,
                "categoria": category
            }


    # ---------------------------------------------------------
    # 🚀 NUEVO: generar múltiples eventos
    # ---------------------------------------------------------
    def generate_events(self, prompt: str, num_events: int = 5):
        """
        Genera varios eventos a partir de un prompt general.
        Devuelve una lista de:
        {
            "titulo": "...",
            "descripcion": "...",
            "categoria": "generado"
        }
        """

        system_prompt = f"""
        Genera {num_events} EVENTOS diferentes en formato JSON.
        Debe ser un arreglo así:
        [
            {{
                "titulo": "...",
                "descripcion": "...",
                "categoria": "..."
            }},
            ...
        ]
        Cada evento debe ser variado, realista y útil.
        Responde SOLO el JSON.
        """

        response = generate_text(system_prompt + "\n\n" + prompt, max_tokens=600)

        if not response:
            return []

        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            json_text = response[start:end]
            events = json.loads(json_text)
            return events
        except Exception as e:
            print("[ERROR] No se pudo parsear el JSON:", e)
            return []
