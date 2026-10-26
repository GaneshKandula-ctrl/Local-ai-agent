import ollama

def main():
    conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        conversation_history.append({"role": "user", "content": user_input})
        model = 'llama3.2'
        response = ollama.chat(model=model, messages=conversation_history)
        answer = response['message']['content']
        conversation_history.append({"role": "assistant", "content": answer})
        print("Chatbot:", answer)

if __name__ == "__main__":
    main()