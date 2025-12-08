from client import get_llama_client

class ChatService:
    def __init__(self):
        self.client = get_llama_client()
        self.history = []

    def send_message(self, user_message: str) -> str:
        # Guardar mensaje del usuario
        self.history.append({"role": "user", "content": user_message})

        # Llamada al modelo
        response = self.client.chat_completion(
            messages=self.history,
            max_tokens=200
        )

        ai_message = response.choices[0].message["content"]

        # Guardar en historial
        self.history.append({"role": "assistant", "content": ai_message})

        return ai_message