from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.settings import ModelSettings
model = OllamaModel("llama3.2")
SYSTEM_PROMPT = """You are a helpful personal desktop assistant.
OPERATIONAL RULES:
1. Always respond with normal conversational text by default (e.g., for greetings, general questions, explanations).
2. DO NOT call any tool unless the user explicitly instructs you to perform an action (e.g., "save a note", "check status", "create file").
3. If the user just says "Hello", "How are you?", or asks an informational question, answer directly in plain text without invoking tools.
"""
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=0.1)
)

@agent.tool_plain
def create_note(title: str, content: str) -> str:
    """
    Saves a text note locally.
    
    CRITICAL: Only call this tool if the user explicitly asks to 'save a note', 
    'take a note', or 'write down' specific content. Do NOT call this for greetings 
    or answers to general questions.
    """
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