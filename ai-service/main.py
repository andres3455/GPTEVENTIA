from services.chat_service import ChatService

def start_console_chat():
    print("=== Chat con LLaMA (escribe 'salir' para terminar) ===\n")

    chat = ChatService()

    while True:
        user_input = input("Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Chat finalizado.")
            break

        response = chat.send_message(user_input)
        print(f"LLaMA: {response}\n")


if __name__ == "__main__":
    start_console_chat()