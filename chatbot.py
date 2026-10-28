from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
import os
import shutil
from pathlib import Path
model = OllamaModel("qwen3.5:4b")
SYSTEM_PROMPT = (
    "You are a helpful personal desktop assistant which can perform tasks like answering questions, reading, saving and moving files."
)
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)

KNOWN_FOLDERS = {
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "desktop": Path.home() / "Desktop",
    "workspace": Path.cwd(),
}

ALLOWED_ROOTS = [
    Path.home(),
    Path.cwd(),
]

def resolve_safe_path(target: str, default_folder: Path = None) -> Path:
    """
    Resolves folder aliases or paths and ensures the target resides 
    within designated safe drive/folder boundaries.
    """
    print(f"Resolving path for target: {target}")
    target_clean = target.strip().lower()
    target_clean = target_clean.split("/",1)
    print(f"Cleaned target: {target_clean}")
    # Check alias first (e.g., 'documents', 'downloads')
    if target_clean[0] in KNOWN_FOLDERS:
        return KNOWN_FOLDERS[target_clean[0]].joinpath(*target_clean[1:]).resolve()

    path = Path(target).expanduser()

    # If it's just a raw filename (e.g. 'notes.txt') and a default folder is given
    if not path.is_absolute() and default_folder:
        path = default_folder / path

    resolved = path.resolve()

    # Verify that the path belongs to at least one allowed root
    is_safe = any(
        str(resolved).lower().startswith(str(root.resolve()).lower())
        for root in ALLOWED_ROOTS
    )

    if not is_safe:
        raise PermissionError(f"Access denied: {resolved} is outside permitted project/user folders.")

    return resolved

@agent.tool_plain
def create_file(filename: str, content: str) -> str:
    """
    Saves a file locally with the specified filename and content.
    Example filename: 'script.py', 'notes.txt', 'index.html'.
    """
    file_path = resolve_safe_path(filename)
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
    print(f"Attempting to read file: {filename}")
    file_path = resolve_safe_path(filename)
    print(f"Resolved file path: {file_path}")
    if not file_path.exists() and not file_path.suffix:
        file_path = file_path.with_suffix(".txt")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of file '{file_path.name}':\n{content}"
    except FileNotFoundError:
        return f"File '{filename}' not found."
    
@agent.tool_plain
def list_folder_contents(folder_path: str = "downloads") -> str:
    """
    Lists all files and subdirectories inside a given folder.
    Accepts aliases like 'downloads', 'documents', 'desktop', or a valid folder path.
    """
    print(f"Attempting to list contents of folder: {folder_path}")
    try:
        path = resolve_safe_path(folder_path)
        print(f"Resolved folder path: {path}")
        if not path.exists() or not path.is_dir():
            return f"Folder '{folder_path}' does not exist or is not a directory."

        items = [f"{'[DIR]' if p.is_dir() else '[FILE]'} {p.name}" for p in path.iterdir()]
        if not items:
            return f"The folder '{path.name}' is empty."

        return f"Contents of {path.name}:\n" + "\n".join(items[:50])  # limit output to 50 items
    except Exception as e:
        return f"Error listing folder: {e}"

@agent.tool_plain
def move_file(source_file_path: str, destination_folder_path: str) -> str:
    """
    Moves an individual file to a target destination folder.
    Creates the destination folder automatically if it doesn't exist.
    """
    print(f"Attempting to move file: {source_file_path}")
    try:
        src = resolve_safe_path(source_file_path)
        dest_folder = resolve_safe_path(destination_folder_path)

        if not src.exists() or not src.is_file():
            return f"Source file '{src.name}' does not exist."

        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / src.name

        # Prevent accidental overwrites
        if dest_file.exists():
            return f"A file named '{src.name}' already exists in '{dest_folder.name}'."

        shutil.move(str(src), str(dest_file))
        return f"Moved '{src.name}' -> '{dest_folder.name}/'."
    except Exception as e:
        return f"Error moving file: {e}"
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