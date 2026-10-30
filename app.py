import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# ---------------------------------------------------------
# 1. PATH SECURITY & WORKSPACE CONFIGURATION
# ---------------------------------------------------------
KNOWN_FOLDERS = {
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "desktop": Path.home() / "Desktop",
    "workspace": Path.cwd(),
}

ALLOWED_ROOTS = [
    Path.home(),
    Path.cwd(),
    Path("D:/Coding"),
]

_gui_logger_callback = None


def log_tool(message: str):
    """Logs tool events to terminal and GUI monitor."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    if _gui_logger_callback:
        _gui_logger_callback(formatted)


def resolve_safe_path(target: str, default_folder: Optional[Path] = None) -> Path:
    log_tool(f"Resolving path: '{target}'")
    target_clean = target.strip().replace("\\", "/").lower()
    parts = target_clean.split("/", 1)

    if parts[0] in KNOWN_FOLDERS:
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
        err = f"Access denied: {resolved} is outside permitted project/user folders."
        log_tool(f"SECURITY ERROR: {err}")
        raise PermissionError(err)

    log_tool(f"Resolved safe path: {resolved}")
    return resolved


# ---------------------------------------------------------
# 2. AGENT SETUP & TOOL DEFINITIONS
# ---------------------------------------------------------
CURRENT_MODEL = "qwen3.5:4b"
provider = OllamaProvider(base_url="http://localhost:11434/v1")
model = OllamaModel(CURRENT_MODEL, provider=provider)

SYSTEM_PROMPT = (
    "You are a helpful personal desktop assistant which can perform tasks like "
    "answering questions, reading, creating, and moving files and folders. "
    "Respond in a friendly, concise manner. Call tools only when explicitly required."
)

agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)


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
        return res
    except Exception as e:
        log_tool(f"Error in create_file: {e}")
        return f"Error creating file: {e}"


@agent.tool_plain
def create_folder(dir_name: str, path_var: str = None) -> str:
    """Creates a directory locally inside the specified folder or workspace."""
    log_tool(f"Tool Invoked: create_folder(name='{dir_name}', parent='{path_var}')")
    try:
        base_dir = resolve_safe_path(path_var) if path_var else Path.cwd()
        final_path = base_dir / dir_name
        final_path.mkdir(parents=True, exist_ok=True)
        res = f"Created directory '{final_path.name}' at {final_path.parent}"
        log_tool(f"Success: {res}")
        return res
    except Exception as e:
        log_tool(f"Error in create_folder: {e}")
        return f"Error: {e}"


@agent.tool_plain
def read_file(filename: str) -> str:
    """Reads a file from local storage using its filename."""
    log_tool(f"Tool Invoked: read_file('{filename}')")
    try:
        file_path = resolve_safe_path(filename)
        if not file_path.exists() and not file_path.suffix:
            file_path = file_path.with_suffix(".txt")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        log_tool(f"Read {len(content)} characters from '{file_path.name}'")
        return f"Content of '{file_path.name}':\n{content}"
    except Exception as e:
        log_tool(f"Error in read_file: {e}")
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
        log_tool(f"Found {len(items)} items in '{path.name}'")
        return f"Contents of {path.name}:\n" + "\n".join(items[:50])
    except Exception as e:
        log_tool(f"Error in list_folder_contents: {e}")
        return f"Error: {e}"


@agent.tool_plain
def move_file(source_file_path: str, destination_folder_path: str) -> str:
    """Moves an individual file to a target destination folder."""
    log_tool(f"Tool Invoked: move_file('{source_file_path}' -> '{destination_folder_path}')")
    try:
        src = resolve_safe_path(source_file_path)
        dest_folder = resolve_safe_path(destination_folder_path)

        if not src.exists() or not src.is_file():
            return f"Source file '{src.name}' does not exist."

        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / src.name

        shutil.move(str(src), str(dest_file))
        res = f"Moved '{src.name}' -> '{dest_folder.name}/'"
        log_tool(f"Success: {res}")
        return res
    except Exception as e:
        log_tool(f"Error in move_file: {e}")
        return f"Error moving file: {e}"


# ---------------------------------------------------------
# 3. MODERN DESKTOP GUI IMPLEMENTATION
# ---------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class LocalAIAssistantGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Local AI Desktop Assistant")
        self.geometry("1150x740")
        self.minsize(900, 600)
        self.configure(fg_color="#181818")

        self.conversation_history = []
        self.is_generating = False

        global _gui_logger_callback
        _gui_logger_callback = self.append_tool_log

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_chat_panel()
        self._build_log_panel()

    def _build_chat_panel(self):
        self.chat_container = ctk.CTkFrame(self, fg_color="#181818", corner_radius=0)
        self.chat_container.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=15)
        self.chat_container.grid_rowconfigure(0, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)

        # Scrollable area for messages
        self.chat_box = ctk.CTkScrollableFrame(
            self.chat_container, fg_color="#181818", corner_radius=0
        )
        self.chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 15))

        # Centered Placeholder Logo
        self.welcome_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        self.welcome_frame.pack(expand=True, pady=120)

        self.logo_label = ctk.CTkLabel(
            self.welcome_frame,
            text="🦙",
            font=ctk.CTkFont(size=56),
            text_color="#ffffff",
            fg_color="#262626",
            width=90,
            height=90,
            corner_radius=45,
        )
        self.logo_label.pack(pady=10)

        self.welcome_text = ctk.CTkLabel(
            self.welcome_frame,
            text="How can I assist your desktop today?",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#888888",
        )
        self.welcome_text.pack()

        # Floating Bottom Input Capsule
        self.input_card = ctk.CTkFrame(
            self.chat_container,
            fg_color="#242424",
            corner_radius=22,
            border_width=1,
            border_color="#333333",
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
            fg_color="#333333",
            hover_color="#444444",
            text_color="#dddddd",
            command=self._on_plus_clicked,
        )
        self.plus_btn.grid(row=0, column=0, padx=(10, 5), pady=8)

        self.entry = ctk.CTkEntry(
            self.input_card,
            placeholder_text="Send a message...",
            placeholder_text_color="#6e6e6e",
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            border_width=0,
            text_color="#ffffff",
        )
        self.entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.model_selector = ctk.CTkOptionMenu(
            self.input_card,
            values=["qwen3.5:4b", "llama3.2", "qwen2.5:3b", "qwen3:8b"],
            font=ctk.CTkFont(size=12),
            fg_color="#333333",
            button_color="#3a3a3a",
            button_hover_color="#4a4a4a",
            text_color="#e0e0e0",
            corner_radius=14,
            width=115,
            height=32,
            command=self._on_model_change,
        )
        self.model_selector.set("qwen3.5:4b")
        self.model_selector.grid(row=0, column=2, padx=(5, 6), pady=8)

        self.send_btn = ctk.CTkButton(
            self.input_card,
            text="↑",
            font=ctk.CTkFont(size=18, weight="bold"),
            width=36,
            height=36,
            corner_radius=18,
            fg_color="#3e3e3e",
            hover_color="#555555",
            text_color="#ffffff",
            command=self.send_message,
        )
        self.send_btn.grid(row=0, column=3, padx=(4, 10), pady=8)

    def _build_log_panel(self):
        self.log_container = ctk.CTkFrame(
            self,
            fg_color="#1e1e1e",
            corner_radius=12,
            border_width=1,
            border_color="#2b2b2b",
        )
        self.log_container.grid(row=0, column=1, sticky="nsew", padx=(5, 15), pady=15)
        self.log_container.grid_rowconfigure(1, weight=1)
        self.log_container.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.log_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header_frame,
            text="🛠️ Tool Call Activity",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#cccccc",
        )
        title.grid(row=0, column=0, sticky="w")

        clear_btn = ctk.CTkButton(
            header_frame,
            text="Clear",
            font=ctk.CTkFont(size=11),
            width=48,
            height=24,
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            text_color="#aaaaaa",
            command=self.clear_tool_logs,
        )
        clear_btn.grid(row=0, column=1, sticky="e")

        self.log_textbox = ctk.CTkTextbox(
            self.log_container,
            fg_color="#141414",
            text_color="#4ec9b0",
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
            corner_radius=8,
            border_width=1,
            border_color="#262626",
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.log_textbox.insert("end", "System ready. Local model listener active.\n")
        self.log_textbox.configure(state="disabled")

    # ---------------------------------------------------------
    # 4. MESSAGE DISPLAY (CLEAN PROSE / NO ASSISTANT BOXES)
    # ---------------------------------------------------------
    def add_user_message(self, text: str):
        """Displays user prompt as a soft rounded capsule on the right side."""
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()

        row_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        row_frame.pack(fill="x", pady=(10, 14), padx=(40, 10), anchor="e")

        bubble = ctk.CTkFrame(
            row_frame,
            fg_color="#262626",
            corner_radius=18,
            border_width=0,
        )
        bubble.pack(anchor="e")

        label = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=14),
            text_color="#e5e5e5",
            wraplength=520,
            justify="left",
        )
        label.pack(padx=16, pady=9)
        self.chat_box._parent_canvas.yview_moveto(1.0)

    def add_assistant_message(self) -> tuple[ctk.CTkLabel, ctk.CTkButton]:
        """Displays assistant response as borderless, clean text on the left side."""
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()

        msg_frame = ctk.CTkFrame(self.chat_box, fg_color="transparent")
        msg_frame.pack(fill="x", pady=(6, 16), padx=(10, 30), anchor="w")

        # Plain text label directly on canvas (no background, no box)
        text_label = ctk.CTkLabel(
            msg_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="#dedede",
            wraplength=600,
            justify="left",
            anchor="w",
        )
        text_label.pack(anchor="w", fill="x", padx=4, pady=(2, 6))

        # Action bar underneath with a subtle copy button
        action_bar = ctk.CTkFrame(msg_frame, fg_color="transparent")
        action_bar.pack(anchor="w", padx=4)

        copy_btn = ctk.CTkButton(
            action_bar,
            text="📋",
            font=ctk.CTkFont(size=12),
            width=28,
            height=24,
            fg_color="transparent",
            hover_color="#262626",
            text_color="#777777",
            corner_radius=6,
        )
        # Store callback hook once text finishes generating
        copy_btn.configure(command=lambda: self._copy_to_clipboard(text_label.cget("text"), copy_btn))
        copy_btn.pack(side="left")

        self.chat_box._parent_canvas.yview_moveto(1.0)
        return text_label, copy_btn

    def _copy_to_clipboard(self, content: str, button: ctk.CTkButton):
        self.clipboard_clear()
        self.clipboard_append(content)
        button.configure(text="✓")
        self.after(1500, lambda: button.configure(text="📋"))

    # ---------------------------------------------------------
    # 5. EVENT HANDLERS & INFERENCE WORKER
    # ---------------------------------------------------------
    def _on_plus_clicked(self):
        log_tool("Action: Opened quick actions menu.")

    def _on_model_change(self, selected_model: str):
        global agent
        log_tool(f"Switching agent model to '{selected_model}'...")
        new_model = OllamaModel(selected_model, provider=provider)
        agent = Agent(model=new_model, system_prompt=SYSTEM_PROMPT)
        log_tool(f"Model successfully switched to {selected_model}.")

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

    def send_message(self):
        if self.is_generating:
            return

        user_text = self.entry.get().strip()
        if not user_text:
            return

        self.entry.delete(0, "end")
        self.add_user_message(user_text)

        # Create borderless left-aligned label for streaming assistant reply
        response_label, copy_btn = self.add_assistant_message()
        self.is_generating = True
        self.send_btn.configure(state="disabled")

        threading.Thread(
            target=self._run_inference_worker,
            args=(user_text, response_label),
            daemon=True,
        ).start()

    def _run_inference_worker(self, user_text: str, label_widget: ctk.CTkLabel):
        accumulated_text = ""
        try:
            with agent.run_stream_sync(
                user_text, message_history=self.conversation_history
            ) as result:
                for chunk in result.stream_text(delta=True):
                    accumulated_text += chunk
                    self.after(
                        0,
                        lambda txt=accumulated_text: (
                            label_widget.configure(text=txt),
                            self.chat_box._parent_canvas.yview_moveto(1.0),
                        ),
                    )

                self.conversation_history = result.all_messages()

        except Exception as e:
            err_msg = f"[Error]: {e}"
            log_tool(f"Inference error: {e}")
            self.after(0, lambda: label_widget.configure(text=err_msg))

        finally:
            self.is_generating = False
            self.after(0, lambda: self.send_btn.configure(state="normal"))


if __name__ == "__main__":
    app = LocalAIAssistantGUI()
    app.mainloop()