from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

model = OllamaModel("qwen3.5:4b")
SYSTEM_PROMPT = (
    "You are a helpful personal desktop assistant which can perform tasks like saving files, reading files, and answering questions."
)
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)

@agent.tool_plain
def create_file(filename: str, content: str) -> str:
    """
    Saves a file locally with the specified filename and content.
    Example filename: 'script.py', 'notes.txt', 'index.html'.
    """
    file_path = Path(filename)
    if not file_path.suffix:
        file_path = file_path.with_suffix(".txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Successfully saved file '{file_path.name}' with {len(content)} characters."

@agent.tool_plain
def read_file(filename: str) -> str:
    """
    Reads a file from local storage using its filename.
    Example filename: 'script.py', 'notes.txt'.
    """
    file_path = Path(filename)
    if not file_path.exists() and not file_path.suffix:
        file_path = file_path.with_suffix(".txt")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of file '{file_path.name}':\n{content}"
    except FileNotFoundError:
        return f"File '{filename}' not found."

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

        print("\nAssistant: ", end="", flush=True)
        try:
            with agent.run_stream_sync(user_input, message_history=conversation_history) as result:
                for chunk in result.stream_text(delta=True):
                    print(chunk, end="", flush=True)
                print()
            conversation_history = result.all_messages()
        except Exception as e:
            print(f"\nError: {e}")
            print("Ensure Ollama is running (`ollama serve`).")

if __name__ == "__main__":
    main()