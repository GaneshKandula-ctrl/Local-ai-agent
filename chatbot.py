from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.settings import ModelSettings
model = OllamaModel("qwen3.5:4b")
SYSTEM_PROMPT = "You are a helpful personal desktop assistant which can perform tasks like saving files, reading files, and answering questions."
# OPERATIONAL RULES:
# 1. Always respond with normal conversational text by default (e.g., for greetings, general questions, explanations).
# 2. DO NOT call any tool unless the user explicitly instructs you to perform an action (e.g., "save a note", "check status", "create file").
# 3. If the user just says "Hello", "How are you?", or asks an informational question, answer directly in plain text without invoking tools.

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)

@agent.tool_plain
def create_file(title: str, content: str) -> str:
    """
    Saves a file or note locally. Use this tool only when the user explicitly asks to 'save a file' or 'create a note' with specific content.
    """
    with open(f"{title}.txt", "w") as f:
        f.write(content)
    return f"Successfully saved file '{title}' with {len(content)} characters."

@agent.tool_plain
def read_file(title: str) -> str:
    """
    Reads a file or note from local storage. Use this tool only when the user explicitly asks to 'read a file' with a specific title.
    """
    try:
        with open(f"{title}.txt", "r") as f:
            content = f.read()
        return f"Content of file '{title}':\n{content}"
    except FileNotFoundError:
        return f"File '{title}' not found."

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
            with agent.run_stream_sync(user_input,message_history=conversation_history) as result:
                for chunk in result.stream_text(delta=True):
                    print(chunk, end="", flush=True)
                print()
            conversation_history = result.all_messages()
        except Exception as e:
            print(f"\nError: {e}")
            print("Ensure Ollama is running (`ollama serve`).")

if __name__ == "__main__":
    main()