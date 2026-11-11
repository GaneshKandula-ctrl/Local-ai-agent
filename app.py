import json
import os
import platform
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import psutil
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# ---------------------------------------------------------
# 1. DIRECTORY, SECURITY & WORKSPACE SETTINGS
# ---------------------------------------------------------
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_MESSAGES = 10  # Sliding window: keeps the latest 10 messages

KNOWN_FOLDERS = {
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "desktop": Path.home() / "Desktop",
}

ALLOWED_ROOTS = [
    Path.home(),
    Path.cwd(),
    Path("D:/Coding"),
]

_gui_logger_callback = None
_gui_confirm_callback = None


def log_tool(message: str):
    """Logs tool events to terminal and GUI monitor."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    if _gui_logger_callback:
        _gui_logger_callback(formatted)


def request_user_confirmation(action_description: str) -> bool:
    """Thread-safe hook to request confirmation from the GUI before destructive operations."""
    if _gui_confirm_callback:
        return _gui_confirm_callback(action_description)
    return True


def resolve_safe_path(target: str, default_folder: Optional[Path] = Path.home() / "Desktop") -> Path:
    log_tool(f"Resolving path: '{target}'")
    target_clean = target.strip().replace("\\", "/").lower()
    parts = target_clean.split("/", 1)
    if not parts[0]:
        parts = parts[1:]

    if parts and parts[0] in KNOWN_FOLDERS:
        resolved = KNOWN_FOLDERS[parts[0]]
        if len(parts) > 1 and parts[1]:
            resolved = resolved / parts[1]
        resolved = resolved.resolve()
        log_tool(f"Mapped alias '{parts[0]}' -> {resolved}")
        return resolved

    path = Path(target).expanduser()
    if not path.is_absolute() and default_folder:
        path = default_folder / path

    resolved = path.resolve()

    is_safe = any(
        str(resolved).lower().startswith(str(root.resolve()).lower())
        for root in ALLOWED_ROOTS
    )

    if not is_safe:
        err = f"Access denied: '{target}' is outside permitted directories."
        log_tool(f"Security error: {err}")
        raise PermissionError(err)

    log_tool(f"Resolved safe path: {resolved}")
    return resolved


def truncate_context_window(messages: list, max_messages: int = MAX_CONTEXT_MESSAGES) -> list:
    """
    Applies a sliding window to the conversation history to prevent context overflow
    while finding a clean turn boundary.
    """
    if len(messages) <= max_messages:
        return messages

    # Slice the last max_messages
    start_idx = len(messages) - max_messages
    # Advance to the first user/request message to maintain valid call/return pairs
    for i in range(start_idx, len(messages)):
        msg = messages[i]
        if getattr(msg, "kind", None) == "request" or hasattr(msg, "parts"):
            start_idx = i
            break

    truncated = messages[start_idx:]
    log_tool(f"Context window truncated from {len(messages)} to {len(truncated)} messages.")
    return truncated


# ---------------------------------------------------------
# 2. AGENT SETUP & TOOL DEFINITIONS
# ---------------------------------------------------------
CURRENT_MODEL = "qwen3.5:4b"
provider = OllamaProvider(base_url="http://localhost:11434/v1")
model = OllamaModel(CURRENT_MODEL, provider=provider)

SYSTEM_PROMPT = (
    "You are a helpful personal desktop assistant that can answer questions, "
    "handle files and folders, and inspect system resources. "
    "Respond in a friendly, concise manner. Call tools only when explicitly required."
)

agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)


@agent.tool_plain
def get_current_time():
    """Returns the current local time."""
    log_tool("Tool Invoked: get_current_time()")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@agent.tool_plain
def create_file(filename: str, content: str) -> str:
    """Creates/saves a file locally with the specified filename and content."""
    log_tool(f"Tool Invoked: create_file(filename='{filename}')")
    try:
        file_path = resolve_safe_path(filename)
        if not file_path.suffix:
            file_path = file_path.with_suffix(".txt")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        res = f"Saved '{file_path.name}' ({len(content)} chars) in {file_path.parent}"
        log_tool(f"Success: {res}")
        return f"Successfully created '{filename}' at {file_path.parent}."
    except Exception as e:
        log_tool(f"Error in create_file: {e}")
        return "Error creating file."


@agent.tool_plain
def launch_application(app_name: str) -> str:
    """Opens a desktop application by name (e.g. 'notepad', 'calculator', 'chrome')."""
    log_tool(f"Tool Invoked: launch_application(app_name='{app_name}')")
    current_os = platform.system().lower()
    try:
        if current_os == "windows":
            win_apps = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "edge": "msedge.exe",
            }
            target = win_apps.get(app_name.lower(), app_name)
            subprocess.Popen(target, shell=True)
        elif current_os == "darwin":
            subprocess.Popen(["open", "-a", app_name])
        else:
            subprocess.Popen([app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        log_tool(f"Successfully opened '{app_name}'")
        return f"Successfully launched '{app_name}'."
    except Exception as e:
        log_tool(f"Error in launch_application: {e}")
        return f"Failed to open '{app_name}': {str(e)}"


@agent.tool_plain
def get_resource_usage() -> str:
    """Returns the current CPU and RAM usage."""
    log_tool("Tool Invoked: get_resource_usage()")
    try:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        gb = 1024**3
        return (
            f"Resource Stats:\n"
            f"- CPU Usage: {cpu_pct}%\n"
            f"- RAM Usage: {memory.percent}%\n"
            f"- Available RAM: {memory.available / gb:.2f} GB / {memory.total / gb:.2f} GB"
        )
    except Exception as e:
        return f"Error retrieving resource usage: {str(e)}"


@agent.tool_plain
def list_running_processes(limit: int = 10) -> str:
    """Lists top running processes sorted by memory usage."""
    log_tool(f"Tool Invoked: list_running_processes(limit={limit})")
    try:
        processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        sorted_procs = sorted(processes, key=lambda x: x["memory_percent"] or 0, reverse=True)[:limit]
        output = [f"{'PID':<8} {'Process Name':<25} {'Memory %':<10}", "-" * 45]
        for p in sorted_procs:
            output.append(f"{p['pid']:<8} {p['name']:<25} {p['memory_percent']:.2f}%")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing processes: {str(e)}"


@agent.tool_plain
def close_application(app_name: str) -> str:
    """
    Finds and closes running application instances.
    Requires user confirmation before terminating.
    """
    log_tool(f"Tool Invoked: close_application(app_name='{app_name}')")

    # Safety Guardrail: Prompt user before closing
    if not request_user_confirmation(f"Close application '{app_name}'? Any unsaved work in it may be lost."):
        log_tool(f"User rejected closing '{app_name}'.")
        return f"Action cancelled: User denied permission to close '{app_name}'."

    closed_any = False
    win_apps = {"notepad": "notepad.exe", "calculator": "CalculatorApp.exe", "chrome": "chrome.exe", "edge": "msedge.exe"}
    target = win_apps.get(app_name.lower(), app_name)

    for process in psutil.process_iter(["pid", "name"]):
        try:
            if process.info["name"].lower() == target.lower():
                process.terminate()
                closed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if closed_any:
        log_tool(f"Terminated {app_name}.")
        return f"Termination signal sent to {app_name}."
    return f"'{app_name}' was not running."


@agent.tool_plain
def create_folder(dir_name: str, path_var: str = None) -> str:
    """Creates a directory locally inside the specified folder or workspace."""
    log_tool(f"Tool Invoked: create_folder(name='{dir_name}', parent='{path_var}')")
    try:
        base_dir = resolve_safe_path(path_var) if path_var else Path.cwd()
        final_path = base_dir / dir_name
        final_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory '{dir_name}' at {final_path.parent}."
    except Exception as e:
        return f"Error creating directory: {e}"


@agent.tool_plain
def read_file(filename: str) -> str:
    """Reads a file from local storage."""
    log_tool(f"Tool Invoked: read_file('{filename}')")
    try:
        file_path = resolve_safe_path(filename)
        if not file_path.exists() and not file_path.suffix:
            file_path = file_path.with_suffix(".txt")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of '{filename}':\n{content}"
    except Exception as e:
        return f"Error reading file '{filename}': {e}"


@agent.tool_plain
def list_folder_contents(folder_path: str = "downloads") -> str:
    """Lists files and directories inside a given folder."""
    log_tool(f"Tool Invoked: list_folder_contents('{folder_path}')")
    try:
        path = resolve_safe_path(folder_path)
        if not path.exists() or not path.is_dir():
            return f"Folder '{folder_path}' does not exist."

        items = [f"{'[DIR]' if p.is_dir() else '[FILE]'} {p.name}" for p in path.iterdir()]
        return f"Contents of '{folder_path}':\n" + "\n".join(items[:50])
    except Exception as e:
        return f"Error listing contents of '{folder_path}': {e}"


@agent.tool_plain
def move_file(source_file_path: str, destination_folder_path: str) -> str:
    """
    Moves an individual file to a target destination folder.
    Requires user confirmation before moving.
    """
    log_tool(f"Tool Invoked: move_file('{source_file_path}' -> '{destination_folder_path}')")
    try:
        src = resolve_safe_path(source_file_path)
        dest_folder = resolve_safe_path(destination_folder_path)

        if not src.exists() or not src.is_file():
            return f"Source file '{src.name}' does not exist."

        # Safety Guardrail: Prompt user before moving files
        prompt_text = f"Move '{src.name}' from '{src.parent.name}' to '{dest_folder.name}'?"
        if not request_user_confirmation(prompt_text):
            log_tool(f"User rejected moving '{src.name}'.")
            return f"Action cancelled: User denied permission to move '{src.name}'."

        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / src.name
        shutil.move(str(src), str(dest_file))
        return f"Successfully moved '{src.name}' to '{dest_folder.name}'."
    except Exception as e:
        return f"Error moving file: {e}"


# ---------------------------------------------------------
# 3. MODAL CONFIRMATION DIALOG (HUMAN-IN-THE-LOOP)
# ---------------------------------------------------------
class ConfirmationModal(ctk.CTkToplevel):
    def __init__(self, parent, message: str, colors: dict):
        super().__init__(parent)
        self.title("Action Confirmation")
        self.geometry("420x210")
        self.resizable(False, False)
        self.configure(fg_color=colors["panel"])
        self.transient(parent)
        self.grab_set()

        self.confirmed = False

        # Center dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 210
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 105
        self.geometry(f"+{x}+{y}")

        title_lbl = ctk.CTkLabel(
            self,
            text="⚠️ Confirmation Required",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#e67e22",
        )
        title_lbl.pack(pady=(18, 8))

        msg_lbl = ctk.CTkLabel(
            self,
            text=message,
            font=ctk.CTkFont(size=13),
            text_color=colors["text"],
            wraplength=360,
            justify="center",
        )
        msg_lbl.pack(padx=20, pady=(0, 20))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 15))
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        deny_btn = ctk.CTkButton(
            btn_frame,
            text="Deny",
            font=ctk.CTkFont(size=13),
            fg_color="#c0392b",
            hover_color="#962d22",
            text_color="#ffffff",
            command=self._on_deny,
        )
        deny_btn.grid(row=0, column=0, padx=8, sticky="ew")

        allow_btn = ctk.CTkButton(
            btn_frame,
            text="Allow",
            font=ctk.CTkFont(size=13),
            fg_color="#27ae60",
            hover_color="#1e8449",
            text_color="#ffffff",
            command=self._on_allow,
        )
        allow_btn.grid(row=0, column=1, padx=8, sticky="ew")

    def _on_allow(self):
        self.confirmed = True
        self.destroy()

    def _on_deny(self):
        self.confirmed = False
        self.destroy()


# ---------------------------------------------------------
# 4. MAIN APPLICATION GUI
# ---------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class LocalAIAssistantGUI(ctk.CTk):
    THEME_COLORS = {
        "Light": {
            "window": "#ffffff",
            "chat": "#ffffff",
            "sidebar": "#f8f9fa",
            "sidebar_border": "#e9ecef",
            "input": "#f4f4f4",
            "input_border": "#e5e5e5",
            "button": "#e5e5e5",
            "button_hover": "#d9d9d9",
            "text": "#212121",
            "muted_text": "#6b6b6b",
            "placeholder": "#8a8a8a",
            "logo": "#f0f0f0",
            "user_bubble": "#f4f4f4",
            "panel": "#f7f7f8",
            "panel_border": "#e5e5e5",
            "log": "#ffffff",
            "log_text": "#16806a",
            "copy_hover": "#eeeeee",
            "copy_text": "#777777",
            "session_active": "#e3e3e3",
        },
        "Dark": {
            "window": "#181818",
            "chat": "#181818",
            "sidebar": "#161616",
            "sidebar_border": "#242424",
            "input": "#242424",
            "input_border": "#333333",
            "button": "#333333",
            "button_hover": "#444444",
            "text": "#e5e5e5",
            "muted_text": "#cccccc",
            "placeholder": "#6e6e6e",
            "logo": "#262626",
            "user_bubble": "#262626",
            "panel": "#1e1e1e",
            "panel_border": "#2b2b2b",
            "log": "#141414",
            "log_text": "#4ec9b0",
            "copy_hover": "#262626",
            "copy_text": "#777777",
            "session_active": "#2b2b2b",
        },
    }

    def __init__(self):
        super().__init__()

        self.title("Local AI Desktop Assistant")
        self.geometry("1260x760")
        self.minsize(980, 620)
        self.theme_mode = "Dark"
        self.colors = self.THEME_COLORS[self.theme_mode]
        self.configure(fg_color=self.colors["window"])

        # Session and Generation State
        self.active_session_id = str(uuid.uuid4())
        self.sessions_metadata: dict[str, dict] = {}
        self.conversation_history = []
        self.ui_messages = []
        self.is_generating = False
        self.abort_event = threading.Event()

        self.user_message_widgets: list[tuple[ctk.CTkFrame, ctk.CTkLabel]] = []
        self.assistant_message_widgets: list[tuple[ctk.CTkLabel, ctk.CTkButton]] = []
        self.session_buttons: dict[str, ctk.CTkButton] = {}
        self.chat_row_widgets: list[ctk.CTkFrame] = []
        # Set thread hooks
        global _gui_logger_callback, _gui_confirm_callback
        _gui_logger_callback = self.append_tool_log
        _gui_confirm_callback = self.ask_confirmation

        # Master Grid: Col 0 = Sidebar, Col 1 = Chat, Col 2 = Tool Logs
        self.grid_columnconfigure(0, weight=0, minsize=210)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1, minsize=260)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_chat_panel()
        self._build_log_panel()

        # Load persisted sessions
        self._load_saved_sessions_list()

    # ---------------------------------------------------------
    # UI BUILDERS
    # ---------------------------------------------------------
    def _build_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(
            self,
            fg_color=self.colors["sidebar"],
            corner_radius=0,
            border_width=1,
            border_color=self.colors["sidebar_border"],
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(1, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        # Top "+ New Chat" Button
        self.new_chat_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="+  New Chat",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            corner_radius=19,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=self.create_new_session,
        )
        self.new_chat_btn.grid(
            row=0, column=0, padx=14, pady=(16, 12), sticky="ew"
        )

        # --- CHANGE HERE: Use self.colors["sidebar"] instead of "transparent" ---
        self.sessions_scroll = ctk.CTkScrollableFrame(
            self.sidebar_frame, fg_color=self.colors["sidebar"], corner_radius=0
        )
        self.sessions_scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.sessions_scroll.grid_columnconfigure(0, weight=1)

        # Bottom Theme & Status Button
        self.theme_button = ctk.CTkButton(
            self.sidebar_frame,
            text=(
                "Switch to Light mode"
                if self.theme_mode == "Dark"
                else "Switch to dark mode"
            ),
            font=ctk.CTkFont(size=12),
            height=32,
            corner_radius=16,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=self._toggle_theme,
        )
        self.theme_button.grid(row=2, column=0, padx=14, pady=14, sticky="ew")
    
    def _build_chat_panel(self):
        self.chat_container = ctk.CTkFrame(self, fg_color=self.colors["chat"], corner_radius=0)
        self.chat_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=15)
        self.chat_container.grid_rowconfigure(0, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)

        self.chat_box = ctk.CTkScrollableFrame(
            self.chat_container, fg_color=self.colors["chat"], corner_radius=0
        )
        self.chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

        # Centered Welcome Frame
        self.welcome_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        self.welcome_frame.pack(expand=True, pady=120)

        self.logo_label = ctk.CTkLabel(
            self.welcome_frame,
            text="🦙",
            font=ctk.CTkFont(size=56),
            text_color=self.colors["text"],
            fg_color=self.colors["logo"],
            width=90,
            height=90,
            corner_radius=45,
        )
        self.logo_label.pack(pady=10)

        self.welcome_text = ctk.CTkLabel(
            self.welcome_frame,
            text="How can I assist your desktop today?",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=self.colors["muted_text"],
        )
        self.welcome_text.pack()

        # Floating Bottom Input Card
        self.input_card = ctk.CTkFrame(
            self.chat_container,
            fg_color=self.colors["input"],
            corner_radius=22,
            border_width=1,
            border_color=self.colors["input_border"],
        )
        self.input_card.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.input_card.grid_columnconfigure(1, weight=1)

        self.plus_btn = ctk.CTkButton(
            self.input_card,
            text="+",
            font=ctk.CTkFont(size=18),
            width=36,
            height=36,
            corner_radius=18,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=lambda: log_tool("Quick actions clicked."),
        )
        self.plus_btn.grid(row=0, column=0, padx=(10, 4), pady=8)

        self.entry = ctk.CTkEntry(
            self.input_card,
            placeholder_text="Send a message...",
            placeholder_text_color=self.colors["placeholder"],
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            border_width=0,
            text_color=self.colors["text"],
        )
        self.entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self.entry.bind("<Return>", lambda event: self.handle_action_button())

        self.model_selector = ctk.CTkOptionMenu(
            self.input_card,
            values=["qwen3.5:4b", "llama3.2", "qwen2.5:3b", "qwen3:8b"],
            font=ctk.CTkFont(size=12),
            fg_color=self.colors["button"],
            button_color=self.colors["button"],
            button_hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            corner_radius=14,
            width=115,
            height=32,
            command=self._on_model_change,
        )
        self.model_selector.set("qwen3.5:4b")
        self.model_selector.grid(row=0, column=2, padx=(4, 6), pady=8)

        # Unified Action Button: Sends (↑) or Aborts (■)
        self.action_btn = ctk.CTkButton(
            self.input_card,
            text="↑",
            font=ctk.CTkFont(size=18, weight="bold"),
            width=36,
            height=36,
            corner_radius=18,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=self.handle_action_button,
        )
        self.action_btn.grid(row=0, column=3, padx=(4, 10), pady=8)

    def _build_log_panel(self):
        self.log_container = ctk.CTkFrame(
            self,
            fg_color=self.colors["panel"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["panel_border"],
        )
        self.log_container.grid(row=0, column=2, sticky="nsew", padx=(5, 15), pady=15)
        self.log_container.grid_rowconfigure(1, weight=1)
        self.log_container.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.log_container, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        self.log_title = ctk.CTkLabel(
            header,
            text="🛠️ Tool Activity",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["text"],
        )
        self.log_title.grid(row=0, column=0, sticky="w")

        self.clear_btn = ctk.CTkButton(
            header,
            text="Clear",
            font=ctk.CTkFont(size=11),
            width=46,
            height=24,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["muted_text"],
            command=self.clear_tool_logs,
        )
        self.clear_btn.grid(row=0, column=1, sticky="e")

        self.log_textbox = ctk.CTkTextbox(
            self.log_container,
            fg_color=self.colors["log"],
            text_color=self.colors["log_text"],
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            corner_radius=8,
            border_width=1,
            border_color=self.colors["panel_border"],
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.log_textbox.insert("end", "System ready. Local model listener active.\n")
        self.log_textbox.configure(state="disabled")

    # ---------------------------------------------------------
    # MULTI-SESSION PERSISTENCE & SIDEBAR CONTROLS
    # ---------------------------------------------------------
    def create_new_session(self):
        """Initializes a new empty chat session."""
        if self.is_generating:
            return

        self._save_current_session()
        self.active_session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.ui_messages = []

        # Destroy ONLY message rows, keeping welcome_frame intact
        for frame in self.chat_row_widgets:
            frame.destroy()
        self.chat_row_widgets.clear()
        self.user_message_widgets.clear()
        self.assistant_message_widgets.clear()

        # Re-show the welcome frame if it was hidden
        if self.welcome_frame.winfo_exists() and not self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack(expand=True, pady=120)

        self._refresh_session_sidebar_highlight()
        log_tool(f"Created new session: {self.active_session_id[:8]}")
    def switch_session(self, session_id: str):
        """Loads and restores a session from disk."""
        if self.is_generating or session_id == self.active_session_id:
            return

        self._save_current_session()
        self.active_session_id = session_id
        session_file = SESSIONS_DIR / f"{session_id}.json"

        if not session_file.exists():
            return

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "messages_json" in data and data["messages_json"]:
                self.conversation_history = ModelMessagesTypeAdapter.validate_json(
                    data["messages_json"].encode("utf-8")
                )
            else:
                self.conversation_history = []

            # Destroy ONLY chat rows safely
            for frame in self.chat_row_widgets:
                frame.destroy()
            self.chat_row_widgets.clear()
            self.user_message_widgets.clear()
            self.assistant_message_widgets.clear()

            self.ui_messages = data.get("ui_messages", [])

            if self.ui_messages:
                if self.welcome_frame.winfo_exists() and self.welcome_frame.winfo_ismapped():
                    self.welcome_frame.pack_forget()

                for item in self.ui_messages:
                    if item["role"] == "user":
                        self._render_user_bubble(item["text"])
                    elif item["role"] == "assistant":
                        lbl, btn = self._render_assistant_block()
                        lbl.configure(text=item["text"])
            else:
                if self.welcome_frame.winfo_exists() and not self.welcome_frame.winfo_ismapped():
                    self.welcome_frame.pack(expand=True, pady=120)

            self._refresh_session_sidebar_highlight()
            log_tool(f"Loaded session '{data.get('title', session_id[:8])}'")

        except Exception as e:
            log_tool(f"Error loading session: {e}")

    def _save_current_session(self):
        """Saves current state and conversation messages to a JSON file."""
        if not self.ui_messages and not self.conversation_history:
            return

        title = self.sessions_metadata.get(self.active_session_id, {}).get("title")
        if not title:
            # Default title to the first user question
            first_user = next((m["text"] for m in self.ui_messages if m["role"] == "user"), "New Chat")
            title = (first_user[:24] + "...") if len(first_user) > 24 else first_user

        try:
            # Serialize model messages via Pydantic AI's TypeAdapter
            messages_json = (
                ModelMessagesTypeAdapter.dump_json(self.conversation_history).decode("utf-8")
                if self.conversation_history
                else ""
            )

            session_data = {
                "session_id": self.active_session_id,
                "title": title,
                "updated_at": datetime.now().isoformat(),
                "ui_messages": self.ui_messages,
                "messages_json": messages_json,
            }

            with open(SESSIONS_DIR / f"{self.active_session_id}.json", "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)

            self.sessions_metadata[self.active_session_id] = {"title": title}
            self._render_session_item(self.active_session_id, title)
            self._refresh_session_sidebar_highlight()

        except Exception as e:
            log_tool(f"Error saving session: {e}")

    def _load_saved_sessions_list(self):
        """Scans sessions directory and populates sidebar."""
        for file in sorted(SESSIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sid = data["session_id"]
                title = data.get("title", sid[:8])
                self.sessions_metadata[sid] = {"title": title}
                self._render_session_item(sid, title)
            except Exception:
                continue

    def _render_session_item(self, session_id: str, title: str):
        if session_id in self.session_buttons:
            self.session_buttons[session_id].configure(text=f"💬 {title}")
            return

        btn = ctk.CTkButton(
            self.sessions_scroll,
            text=f"💬 {title}",
            font=ctk.CTkFont(size=12),
            height=34,
            anchor="w",
            corner_radius=8,
            fg_color="transparent",
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=lambda sid=session_id: self.switch_session(sid),
        )
        btn.pack(fill="x", pady=2)
        self.session_buttons[session_id] = btn

    def _refresh_session_sidebar_highlight(self):
        for sid, btn in self.session_buttons.items():
            if sid == self.active_session_id:
                btn.configure(fg_color=self.colors["session_active"])
            else:
                btn.configure(fg_color="transparent")

    # ---------------------------------------------------------
    # CONFIRMATION MODAL DISPATCHER
    # ---------------------------------------------------------
    def ask_confirmation(self, action_desc: str) -> bool:
        """Invoked from background tool threads. Halts worker until user answers modal."""
        event = threading.Event()
        result = {"approved": False}

        def _open():
            modal = ConfirmationModal(self, action_desc, self.colors)
            self.wait_window(modal)
            result["approved"] = modal.confirmed
            event.set()

        self.after(0, _open)
        event.wait()
        return result["approved"]

    # ---------------------------------------------------------
    # CHAT RENDERING
    # ---------------------------------------------------------
    def _render_user_bubble(self, text: str):
        if self.welcome_frame.winfo_exists() and self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()

        row_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        row_frame.pack(fill="x", pady=(10, 14), padx=(40, 10), anchor="e")
        self.chat_row_widgets.append(row_frame)  # <--- Track row

        bubble = ctk.CTkFrame(
            row_frame,
            fg_color=self.colors["user_bubble"],
            corner_radius=18,
            border_width=0,
        )
        bubble.pack(anchor="e")

        label = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text"],
            wraplength=520,
            justify="left",
        )
        label.pack(padx=16, pady=9)
        self.user_message_widgets.append((bubble, label))
        self.chat_box._parent_canvas.yview_moveto(1.0)

    def _render_assistant_block(self) -> tuple[ctk.CTkLabel, ctk.CTkButton]:
        if self.welcome_frame.winfo_exists() and self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()

        msg_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        msg_frame.pack(fill="x", pady=(6, 16), padx=(10, 30), anchor="w")
        self.chat_row_widgets.append(msg_frame)  # <--- Track row

        text_label = ctk.CTkLabel(
            msg_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text"],
            wraplength=600,
            justify="left",
            anchor="w",
        )
        text_label.pack(anchor="w", fill="x", padx=4, pady=(2, 6))

        action_bar = ctk.CTkFrame(msg_frame, fg_color="transparent")
        action_bar.pack(anchor="w", padx=4)

        copy_btn = ctk.CTkButton(
            action_bar,
            text="📋",
            font=ctk.CTkFont(size=12),
            width=28,
            height=24,
            fg_color="transparent",
            hover_color=self.colors["copy_hover"],
            text_color=self.colors["copy_text"],
            corner_radius=6,
        )
        copy_btn.configure(command=lambda: self._copy_to_clipboard(text_label.cget("text"), copy_btn))
        copy_btn.pack(side="left")

        self.assistant_message_widgets.append((text_label, copy_btn))
        self.chat_box._parent_canvas.yview_moveto(1.0)
        return text_label, copy_btn
    def _copy_to_clipboard(self, content: str, button: ctk.CTkButton):
        self.clipboard_clear()
        self.clipboard_append(content)
        button.configure(text="✓")
        self.after(1500, lambda: button.configure(text="📋"))

    # ---------------------------------------------------------
    # GENERATION & ABORT CONTROLS
    # ---------------------------------------------------------
    def handle_action_button(self):
        """Handles either sending a prompt or aborting active generation."""
        if self.is_generating:
            self.abort_generation()
        else:
            self.send_message()

    def abort_generation(self):
        """Signals the active worker thread to stop generating."""
        log_tool("Abort requested by user.")
        self.abort_event.set()
        self.action_btn.configure(text="Stopping...", state="disabled")

    def send_message(self):
        if self.is_generating:
            return

        user_text = self.entry.get().strip()
        if not user_text:
            return

        self.entry.delete(0, "end")
        self._render_user_bubble(user_text)
        self.ui_messages.append({"role": "user", "text": user_text})

        # Morph send button into a red stop button
        self.is_generating = True
        self.abort_event.clear()
        self.action_btn.configure(
            text="■",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            text_color="#ffffff",
            state="normal",
        )

        response_label, _ = self._render_assistant_block()

        threading.Thread(
            target=self._run_inference_worker,
            args=(user_text, response_label),
            daemon=True,
        ).start()

    def _run_inference_worker(self, user_text: str, label_widget: ctk.CTkLabel):
        accumulated_text = ""
        try:
            # 1. Apply Context Window Truncation before passing history
            clean_history = truncate_context_window(self.conversation_history)

            # 2. Run streaming execution
            with agent.run_stream_sync(user_text, message_history=clean_history) as result:
                for chunk in result.stream_text(delta=True):
                    # Check if user clicked the Stop button
                    if self.abort_event.is_set():
                        log_tool("Streaming aborted successfully.")
                        accumulated_text += " [Aborted]"
                        break

                    accumulated_text += chunk
                    self.after(
                        0,
                        lambda txt=accumulated_text: (
                            label_widget.configure(text=txt),
                            self.chat_box._parent_canvas.yview_moveto(1.0),
                        ),
                    )

                # Store updated history if not aborted midway through a broken turn
                if not self.abort_event.is_set():
                    self.conversation_history = result.all_messages()

            self.ui_messages.append({"role": "assistant", "text": accumulated_text})
            self._save_current_session()

        except Exception as e:
            err = f"[Error]: {e}"
            log_tool(f"Inference error: {e}")
            self.after(0, lambda: label_widget.configure(text=err))
            self.ui_messages.append({"role": "assistant", "text": err})

        finally:
            self.is_generating = False
            # Restore button to Send state (↑)
            self.after(
                0,
                lambda: self.action_btn.configure(
                    text="↑",
                    fg_color=self.colors["button"],
                    hover_color=self.colors["button_hover"],
                    text_color=self.colors["text"],
                    state="normal",
                ),
            )

    # ---------------------------------------------------------
    # UTILITIES & THEME
    # ---------------------------------------------------------
    def clear_tool_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def append_tool_log(self, text: str):
        def _update():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", text + "\n")
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")

        self.after(0, _update)

    def _on_model_change(self, selected_model: str):
        global agent
        log_tool(f"Switching agent model to '{selected_model}'...")
        new_model = OllamaModel(selected_model, provider=provider)
        agent = Agent(model=new_model, system_prompt=SYSTEM_PROMPT)
        log_tool(f"Model successfully switched to {selected_model}.")

    def _toggle_theme(self):
        # 1. Flip active theme mode
        self.theme_mode = "Dark" if self.theme_mode == "Light" else "Light"
        self.colors = self.THEME_COLORS[self.theme_mode]

        # 2. Update window & sidebar containers FIRST
        self.configure(fg_color=self.colors["window"])
        self.sidebar_frame.configure(
            fg_color=self.colors["sidebar"],
            border_color=self.colors["sidebar_border"],
        )

        # --- KEY FIX 1: Explicitly re-color the sessions scroll container ---
        self.sessions_scroll.configure(fg_color=self.colors["sidebar"])

        # 3. Inform CustomTkinter of appearance mode change
        ctk.set_appearance_mode(self.theme_mode)

        # 4. Update sidebar buttons
        self.new_chat_btn.configure(
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )
        self.theme_button.configure(
            text=f"Switch to {'dark' if self.theme_mode == 'Light' else 'light'} mode",
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )

        # --- KEY FIX 2: Update all existing session items in sidebar ---
        for sid, btn in self.session_buttons.items():
            btn.configure(
                hover_color=self.colors["button_hover"],
                text_color=self.colors["text"],
            )
        self._refresh_session_sidebar_highlight()

        # 5. Update main chat area & input controls
        self.chat_container.configure(fg_color=self.colors["chat"])
        self.chat_box.configure(fg_color=self.colors["chat"])
        self.input_card.configure(
            fg_color=self.colors["input"],
            border_color=self.colors["input_border"],
        )
        self.plus_btn.configure(
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )
        self.entry.configure(
            placeholder_text_color=self.colors["placeholder"],
            text_color=self.colors["text"],
        )
        self.model_selector.configure(
            fg_color=self.colors["button"],
            button_color=self.colors["button"],
            button_hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )
        self.action_btn.configure(
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )

        # 6. Update tool activity panel
        self.log_container.configure(
            fg_color=self.colors["panel"],
            border_color=self.colors["panel_border"],
        )
        self.log_title.configure(text_color=self.colors["text"])
        self.clear_btn.configure(
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["muted_text"],
        )
        self.log_textbox.configure(
            fg_color=self.colors["log"],
            text_color=self.colors["log_text"],
            border_color=self.colors["panel_border"],
        )

        # 7. Update chat messages
        for bubble, label in self.user_message_widgets:
            bubble.configure(fg_color=self.colors["user_bubble"])
            label.configure(text_color=self.colors["text"])

        for label, copy_button in self.assistant_message_widgets:
            label.configure(text_color=self.colors["text"])
            copy_button.configure(
                hover_color=self.colors["copy_hover"],
                text_color=self.colors["copy_text"],
            )

if __name__ == "__main__":
    app = LocalAIAssistantGUI()
    app.mainloop()