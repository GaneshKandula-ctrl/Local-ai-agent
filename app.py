import os
from os import path
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
import base64

# EMAIL CONFIGURATION

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
GMAIL_USE = 0

def get_gmail_client():
    """Checks, refreshes, and returns a valid Gmail service instance."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('gmail', 'v1', credentials=creds)

# ---------------------------------------------------------
# 1. PATH SECURITY & WORKSPACE CONFIGURATION
# ---------------------------------------------------------
KNOWN_FOLDERS = {
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "desktop": Path.home() / "Desktop",

}

_gui_logger_callback = None


def log_tool(message: str):
    """Logs tool events to terminal and GUI monitor."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    if _gui_logger_callback:
        _gui_logger_callback(formatted)


def resolve_safe_path(target: str, default_folder: Optional[Path] = Path.home() / "Desktop") -> Path:
    log_tool(f"Resolving path: '{target}'")
    target_clean = target.strip().replace("\\", "/").lower()
    parts = target_clean.split("/", 1)
    if not parts[0]:
        parts = parts[1:]
    if parts[0] in KNOWN_FOLDERS:
        resolved = KNOWN_FOLDERS[parts[0]]
        if len(parts) > 1 and parts[1]:
            resolved = resolved / parts[1]
        resolved = resolved.resolve()
        log_tool(f"Mapped alias '{parts[0]}' -> {resolved}")
        return resolved

    if len(parts) == 1 and default_folder:
        resolved = default_folder / parts[0]
        log_tool(f"Mapped alias '{parts[0]}' -> {resolved}")
        return resolved.resolve()
    err = f"Access denied: '{target}' is outside allowed directories."
    log_tool(f"Security error: {err}")
    raise PermissionError(err)


# ---------------------------------------------------------
# 2. AGENT SETUP & TOOL DEFINITIONS
# ---------------------------------------------------------
CURRENT_MODEL = "qwen3.5:4b"
provider = OllamaProvider(base_url="http://localhost:11434/v1")
model = OllamaModel(CURRENT_MODEL, provider=provider)

SYSTEM_PROMPT = (
    "You are a helpful personal desktop assistant which can perform tasks like "
    "answering questions, handling files and folders and emails. "
    "Respond in a friendly manner, use emojis to express emotions. Ask and confirm questions if needed, and avoid making assumptions."
    "Call tools only when explicitly required."
)

agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)


@agent.tool_plain
def get_emails(query="label:INBOX", max_results=1):
    """
    Searches for emails and returns a list of dictionaries with content.
    Query examples: 'is:unread', 'from:boss@company.com', 'subject:report'
    """
    log_tool(f"Tool Invoked: get_emails(query = '{query}', max_results = {max_results})")
    try:
        service = get_gmail_client()
        results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        email_list = []
        for msg in messages:
            # Fetch full details for each message
            txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = txt.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract Subject and From fields
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '(No Subject)')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '(Unknown Sender)')
            
            # Extract body text
            body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
            elif 'body' in payload and 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                
            email_list.append({
                "id": msg['id'],
                "from": sender,
                "subject": subject,
                "snippet": txt.get('snippet', ''),
                "body": body.strip()
            })
        log_tool(f"Successfully retrieved emails: {len(email_list)}")
        return email_list
        
    except Exception as e:
        log_tool(f"Error fetching emails: {e}")
        return []
    
@agent.tool_plain
def send_email(to_email, subject, body_text):
    """Creates and sends an email to a specific recipient."""
    log_tool(f"Tool Invoked: send_email(to='{to_email}', subject='{subject}')")
    try:
        service = get_gmail_client()
        message = MIMEText(body_text)
        message['to'] = to_email
        message['subject'] = subject
        
        # Encode the message in base64 string format required by Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        send_request = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        log_tool(f"Email sent successfully! Message ID: {send_request['id']}")
        return f"Email sent successfully! Message ID: {send_request['id']}"
        
    except Exception as e:
        log_tool(f"Error sending email: {e}")
        return f"Failed to send email: {e}"

@agent.tool_plain
def reply_to_email(original_email_id, reply_text):
    """Replies directly to an existing email thread using its original message ID."""
    log_tool(f"Tool Invoked: reply_to_email(original_id='{original_email_id}')")
    try:
        service = get_gmail_client()
        # 1. Fetch the original email to extract necessary thread details
        orig_msg = service.users().messages().get(userId='me', id=original_email_id).execute()
        thread_id = orig_msg['threadId']
        headers = orig_msg['payload']['headers']
        
        # 2. Find who to send the reply back to (the sender of the original email)
        to_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), None)
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        
        if not to_email:
            return "Could not determine recipient from the original email."
            
        # Ensure subject starts with 'Re:' if it doesn't already
        if not subject.lower().startswith('re:'):
            subject = f"Re: {subject}"
            
        # 3. Construct the reply message
        message = MIMEText(reply_text)
        message['to'] = to_email
        message['subject'] = subject
        
        # Mandatory headers to link the email to the original thread
        message['In-Reply-To'] = original_email_id
        message['References'] = original_email_id
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # 4. Send the message attached to the same threadId
        reply_request = service.users().messages().send(
            userId='me',
            body={
                'raw': raw_message,
                'threadId': thread_id
            }
        ).execute()
        log_tool(f"Reply sent successfully! Message ID: {reply_request['id']} under thread: {thread_id}")
        return f"Reply sent successfully under thread: {thread_id}"
        
    except Exception as e:
        log_tool(f"Error replying to email: {e}")
        return f"Failed to send reply: {e}"

@agent.tool_plain
def read_full_email_body(message_id):
    """Fetch the complete text body of a specific email using its message ID."""
    log_tool(f"Tool Invoked: read_full_email_body(message_id='{message_id}')")
    try:
        service = get_gmail_client()
        txt = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        payload = txt.get('payload', {})
        
        # Helper logic to extract body from plain text or multipart emails
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        else:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')    
        log_tool(f"Successfully retrieved full email body for message ID: {message_id}")
        return body
    except Exception as e:
        log_tool(f"Error retrieving full email: {e}")
        return f"Failed to retrieve email body: {e}"

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
        return f"Error creating file."


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
        return f"Successfully created directory '{dir_name}' at {final_path.parent}."
    except Exception as e:
        log_tool(f"Error in create_folder: {e}")
        return f"Error creating directory."


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
        return f"Content of '{filename}':\n{content}"
    except Exception as e:
        log_tool(f"Error in read_file: {e}")
        return f"Error reading file '{filename}'"


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
        return f"Contents of '{folder_path}':\n" + "\n".join(items[:50])
    except Exception as e:
        log_tool(f"Error in list_folder_contents: {e}")
        return f"Error listing contents of folder '{folder_path}'"


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
        return f"Successfully moved '{src.name}' to '{dest_folder}'."
    except Exception as e:
        log_tool(f"Error in move_file: {e}")
        return f"Error moving file '{source_file_path}' to '{destination_folder_path}'"


# ---------------------------------------------------------
# 3. MODERN DESKTOP GUI IMPLEMENTATION
# ---------------------------------------------------------
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class LocalAIAssistantGUI(ctk.CTk):
    THEME_COLORS = {
        "Light": {
            "window": "#ffffff",
            "chat": "#ffffff",
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
        },
        "Dark": {
            "window": "#181818",
            "chat": "#181818",
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
        },
    }

    def __init__(self):
        super().__init__()

        self.title("Local AI Desktop Assistant")
        self.geometry("1150x740")
        self.minsize(900, 600)
        self.theme_mode = "Light"
        self.colors = self.THEME_COLORS[self.theme_mode]
        self.configure(fg_color=self.colors["window"])

        self.conversation_history = []
        self.is_generating = False
        self.user_message_widgets: list[tuple[ctk.CTkFrame, ctk.CTkLabel]] = []
        self.assistant_message_widgets: list[tuple[ctk.CTkLabel, ctk.CTkButton]] = []

        global _gui_logger_callback
        _gui_logger_callback = self.append_tool_log

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_chat_panel()
        self._build_log_panel()

    def _build_chat_panel(self):
        self.chat_container = ctk.CTkFrame(self, fg_color=self.colors["chat"], corner_radius=0)
        self.chat_container.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=15)
        self.chat_container.grid_rowconfigure(1, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=1)

        self.chat_title = ctk.CTkLabel(
            header_frame,
            text="Local AI Assistant",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=self.colors["text"],
        )
        self.chat_title.grid(row=0, column=0, sticky="w", padx=8)

        self.theme_button = ctk.CTkButton(
            header_frame,
            text="Switch to dark mode",
            font=ctk.CTkFont(size=12),
            width=142,
            height=30,
            corner_radius=15,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=self._toggle_theme,
        )
        self.theme_button.grid(row=0, column=1, sticky="e", padx=8)

        # Scrollable area for messages
        self.chat_box = ctk.CTkScrollableFrame(
            self.chat_container, fg_color=self.colors["chat"], corner_radius=0
        )
        self.chat_box.grid(row=1, column=0, sticky="nsew", pady=(0, 15))

        # Centered Placeholder Logo
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
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["muted_text"],
        )
        self.welcome_text.pack()

        # Floating Bottom Input Capsule
        self.input_card = ctk.CTkFrame(
            self.chat_container,
            fg_color=self.colors["input"],
            corner_radius=22,
            border_width=1,
            border_color=self.colors["input_border"],
        )
        self.input_card.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
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
            command=self._on_plus_clicked,
        )
        self.plus_btn.grid(row=0, column=0, padx=(10, 5), pady=8)

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
        self.entry.bind("<Return>", lambda event: self.send_message())

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
        self.model_selector.grid(row=0, column=2, padx=(5, 6), pady=8)

        self.send_btn = ctk.CTkButton(
            self.input_card,
            text="↑",
            font=ctk.CTkFont(size=18, weight="bold"),
            width=36,
            height=36,
            corner_radius=18,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
            command=self.send_message,
        )
        self.send_btn.grid(row=0, column=3, padx=(4, 10), pady=8)

    def _build_log_panel(self):
        self.log_container = ctk.CTkFrame(
            self,
            fg_color=self.colors["panel"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["panel_border"],
        )
        self.log_container.grid(row=0, column=1, sticky="nsew", padx=(5, 15), pady=15)
        self.log_container.grid_rowconfigure(1, weight=1)
        self.log_container.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.log_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header_frame.grid_columnconfigure(0, weight=1)

        self.log_title = ctk.CTkLabel(
            header_frame,
            text="🛠️ Tool Call Activity",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text"],
        )
        self.log_title.grid(row=0, column=0, sticky="w")

        self.clear_btn = ctk.CTkButton(
            header_frame,
            text="Clear",
            font=ctk.CTkFont(size=11),
            width=48,
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
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
            corner_radius=8,
            border_width=1,
            border_color=self.colors["panel_border"],
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.log_textbox.insert("end", "System ready. Local model listener active.\n")
        self.log_textbox.configure(state="disabled")

    def _toggle_theme(self):
        self.theme_mode = "Dark" if self.theme_mode == "Light" else "Light"
        self.colors = self.THEME_COLORS[self.theme_mode]
        ctk.set_appearance_mode(self.theme_mode)
        self.configure(fg_color=self.colors["window"])
        self.chat_container.configure(fg_color=self.colors["chat"])
        self.chat_box.configure(fg_color=self.colors["chat"])
        self.chat_title.configure(text_color=self.colors["text"])
        self.theme_button.configure(
            text=f"Switch to {'dark' if self.theme_mode == 'Light' else 'light'} mode",
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )
        self.logo_label.configure(
            text_color=self.colors["text"],
            fg_color=self.colors["logo"],
        )
        self.welcome_text.configure(text_color=self.colors["muted_text"])
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
        self.send_btn.configure(
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            text_color=self.colors["text"],
        )
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

        for bubble, label in self.user_message_widgets:
            bubble.configure(fg_color=self.colors["user_bubble"])
            label.configure(text_color=self.colors["text"])

        for label, copy_button in self.assistant_message_widgets:
            label.configure(text_color=self.colors["text"])
            copy_button.configure(
                hover_color=self.colors["copy_hover"],
                text_color=self.colors["copy_text"],
            )

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
            text_color=self.colors["text"],
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
            hover_color=self.colors["copy_hover"],
            text_color=self.colors["copy_text"],
            corner_radius=6,
        )
        # Store callback hook once text finishes generating
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