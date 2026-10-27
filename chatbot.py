from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

model = OllamaModel("llama3.2")

agent = Agent(
    model=model,
    system_prompt=(
        "You are an active local desktop assistant. "
        "You can answer questions or call local tools to inspect files."
    )
)

@agent.tool_plain
def create_note(title: str, content: str) -> str:
    """Create a quick local text note with a title and content."""
    with open(f"{title}.txt", "w") as f:
        f.write(content)
    return f"Successfully saved note '{title}' with {len(content)} characters."

def main():
    print("--- Local Assistant Agent Online (type 'exit' or 'quit' to stop) ---")
    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down assistant.")
            break

        try:
            result = agent.run_sync(
                user_input,
                message_history=conversation_history
            )
            
            print(f"\nAssistant: {result.output}")

            conversation_history = result.all_messages()

        except Exception as e:
            print(f"\nError: {e}")
            print("Ensure Ollama is running (`ollama serve`).")

if __name__ == "__main__":
    main()