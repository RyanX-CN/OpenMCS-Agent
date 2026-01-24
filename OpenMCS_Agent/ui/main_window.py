import os
import sys
import re
import html 
import uuid
import datetime

from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer, PythonLexer
from pygments.formatters import HtmlFormatter

from PyQt5.QtWidgets import (
    QApplication,QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QScrollArea, QPushButton, QLabel, QFrame, QGraphicsDropShadowEffect,
    QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(current_dir)
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from config.settings import get_available_models
from core.agent import build_agent
from core.multi_agent import build_multi_agent_graph
from core.schemas import Context
from ui.widgets import ChatInput
from ui.worker import AgentWorker
from ui.code_editor import CodeEditorWindow

from utils.document_loader import load_html, load_pdf, load_json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

HELLO_MESSAGE = """
Hello, I'm a robot assistant to help you explore OpenMCS (Open Microscopy Control Software), an extensible, plugin-based device control framework developed by WeLab for advanced optical microscopy experiments！What can I do include:
- Answer your questions about OpenMCS features and usage.
- Assist you in generating device integration plugins based on provided SDK documentation.
- Provide guidance on integrating microscopy devices.
"""

class OpenMCSChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenMCS-Agent")
        self.setWindowIcon(QIcon(os.path.join(project_root, "resources", "logo", "OpenMCS-Agent.png")))
        self.resize(900, 700)
        
        self.agent = None
        self.agent_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.agent_context = Context(operator_id="gui_user")
        
        self.code_editor = None # Code Editor Window instance

        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._update_loading_animation)
        self.loading_dots = 0

        self.file_paths = []
        self.display_files_text = ""

        self._init_ui()
        
        self.add_message("assistant", HELLO_MESSAGE, is_user=False)
        self.agent = build_agent(self.cbox_model.currentText())
        # self.agent = build_multi_agent_graph(self.cbox_model.currentText()) # raw graph (for debugging)

    def _init_ui(self):
        font = QFont("Arial", 12)
        self.setFont(font)
        self._setup_stylesheet()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 10) 
        main_layout.setSpacing(10)

        # 1. 消息区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.messages_widget = QWidget()
        self.messages_widget.setObjectName("scrollContent")
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(15)
        self.messages_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_area.setWidget(self.messages_widget)
        main_layout.addWidget(self.scroll_area, 1)

        # 2. 工具栏区域 (模型选择 + 文件上传)
        toolbar_container = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(0, 0, 0, 5) # Reduced margins
        toolbar_layout.setSpacing(8) # Reduced spacing

        self.cbox_model = QComboBox()
        self.cbox_model.addItems(get_available_models())
        self.cbox_model.setFixedWidth(120)
        self.cbox_model.setFixedHeight(30)
        self.cbox_model.setStyleSheet("font-size: 12px; font-family: Arial;")
        self.cbox_model.currentIndexChanged.connect(self.on_model_changed)
        
        self.btn_reset = QPushButton("↻")
        self.btn_reset.setToolTip("Reset Session")
        self.btn_reset.setFixedSize(30, 30)
        self.btn_reset.setStyleSheet("font-size: 16px; padding: 0px; border-radius: 5px; background-color: #E74C3C;") 
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        
        self.btn_open_editor = QPushButton("→")
        self.btn_open_editor.setToolTip("Open Code Editor")
        self.btn_open_editor.setFixedSize(30, 30)
        self.btn_open_editor.setStyleSheet("font-size: 16px; padding: 0px; border-radius: 5px; background-color: #28a745;")
        self.btn_open_editor.clicked.connect(lambda: self.open_in_editor({}))

        self.btn_upload_files = QPushButton("+")
        self.btn_upload_files.setToolTip("Upload Documents")
        self.btn_upload_files.setFixedSize(30, 30)
        self.btn_upload_files.setStyleSheet("font-size: 16px; padding: 0px; border-radius: 5px;") 
        self.btn_upload_files.clicked.connect(self.on_upload_clicked)

        self.label_files = QLabel("No file selected")
        self.label_files.setStyleSheet("color: #666; font-style: italic; font-size: 12px;")
        

        toolbar_layout.addWidget(self.cbox_model)
        toolbar_layout.addWidget(self.btn_reset)
        toolbar_layout.addWidget(self.btn_open_editor)
        toolbar_layout.addWidget(self.btn_upload_files)
        toolbar_layout.addWidget(self.label_files)
        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar_container, 0)

        # 3. 输入区域
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(10)

        self.input = ChatInput()
        self.input.sendMessage.connect(self.on_send_clicked)

        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedHeight(60)
        self.send_btn.setFixedWidth(150)
        self.send_btn.clicked.connect(self.on_send_clicked)

        input_layout.addWidget(self.input)
        input_layout.addWidget(self.send_btn)
        main_layout.addWidget(input_container, 0)

    def on_model_changed(self):
        """切换模型时重新构建 Agent"""
        provider = self.cbox_model.currentText()
        try:
            self.agent = build_agent(provider)
            self.add_message("assistant", f"Switched to model provider: **{provider}**", is_user=False)
        except Exception as e:
            self.add_message("assistant", f"Failed to switch model: {str(e)}", is_user=False)

    def on_reset_clicked(self):
        """Reset the chat session and clear history."""
        self.agent_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # Clear UI messages
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Re-add hello message

        self.add_message("assistant", HELLO_MESSAGE, is_user=False)
        self.add_message("assistant", "Session has been reset. Memory cleared.", is_user=False)

    def on_upload_clicked(self):
        """处理多文件上传"""
        self.file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "", 
            "All Files (*.*);;PDF Files (*.pdf);;HTML Files (*.html);;Python Files (*.py)"
        )
        
        if self.file_paths:
            for file in self.file_paths:
                self.display_files_text += f"📎 {os.path.basename(file)} "
            # count = len(self.file_paths)
            # if count == 1:
            #     display_text = f"📎 {os.path.basename(self.file_paths[0])}"
            # else:
            #     display_text = f"📎 {count} files selected"
            self.label_files.setText(self.display_files_text)
            
            loaded_files_info = []

            for file_path in self.file_paths:
                filename = os.path.basename(file_path)
                try:
                    content = ""
                    if file_path.endswith(".pdf"):
                        docs = load_pdf(file_path)
                        content = "\n".join([d.page_content for d in docs])
                    elif file_path.endswith(".html") or file_path.endswith(".htm"):
                        docs = load_html(file_path)
                        content = "\n".join([d.page_content for d in docs])
                    else:
                        # Default text read
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                    # Simple logic to categorize files
                    if file_path.endswith(".py"):
                        self.agent_context.uploaded_framework_files[filename] = content
                        type_str = "Framework"
                    else:
                        self.agent_context.uploaded_sdk_docs[filename] = content
                        type_str = "SDK Doc"
                    
                    loaded_files_info.append(f"{filename} ({type_str})")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            
            # Show summary in chat
            if loaded_files_info:
                info_str = "<br>".join([f"• {info}" for info in loaded_files_info])
                self.add_message("assistant", f"Loaded {len(loaded_files_info)} files:{info_str}", is_user=False)

    def _update_loading_animation(self):
        dots = "." * (self.loading_dots % 4)
        self.send_btn.setText(f"Thinking{dots}")
        self.loading_dots += 1
        
        if hasattr(self, 'waiting_message_label') and self.waiting_message_label:
            elapsed = (datetime.datetime.now() - self.waiting_start_time).total_seconds()
            self.waiting_message_label.setText(f"Thinking... {int(elapsed)}s")

    def _setup_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F0F2F5; }
            QScrollArea { border: none; background-color: transparent; }
            QWidget#scrollContent { background-color: transparent; }
            QScrollBar:vertical { border: none; background: transparent; width: 8px; margin: 0px; }
            QScrollBar::handle:vertical { background: #C1C1C1; min-height: 20px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #A8A8A8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            
            QTextEdit { background-color: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 10px; padding: 10px; font-size: 16px; color: #333333; }
            QTextEdit:focus { border: 1px solid #0078D7; }
            QTextEdit:disabled { background-color: #F5F5F5; color: #888888; }
            
            QPushButton { background-color: #0078D7; color: white; border-radius: 10px; font-weight: bold; font-size: 16px; }
            QPushButton:hover { background-color: #0063B1; }
            QPushButton:pressed { background-color: #004E8C; }
            QPushButton:disabled { background-color: #CCCCCC; }
            
            QComboBox {
                border: 1px solid #E5E5E5;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
                font-size: 14px;
            }
        """)

    def _add_waiting_message(self):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        bubble = QFrame()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        bubble.setGraphicsEffect(shadow)

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 10, 15, 10)
        bubble_layout.setSpacing(0)
        
        self.waiting_message_label = QLabel("Thinking... 0.0s")
        self.waiting_message_label.setStyleSheet("color: #666; font-style: italic; font-size: 14px;")
        bubble_layout.addWidget(self.waiting_message_label)
        
        bubble.setStyleSheet("QFrame { background-color: #FFFFFF; border-radius: 10px; border-top-left-radius: 2px; }")
        
        row_layout.addWidget(bubble)
        row_layout.addStretch()
        
        self.messages_layout.addWidget(row_widget)
        self._scroll_to_bottom()
        
        return row_widget

    def add_message(self, role, text, is_user: bool, files: dict = None):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Parse logic for AI messages to separate role name from content
        display_role = "Assistant"
        display_text = text
        if not is_user:
            # Check for **[RoleName]** pattern at the beginning of the text
            match = re.match(r"^\*\*\[(.*?)\]\*\*\s*", text)
            if match:
                display_role = match.group(1)
                display_text = text[match.end():] # Remove the tag from the displayed text
        
        bubble = QFrame()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        bubble.setGraphicsEffect(shadow)

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 10, 15, 10)
        bubble_layout.setSpacing(0)
        
        parts = re.split(r'(```.*?```)', display_text, flags=re.DOTALL)
        final_html_parts = []
        
        detected_code_files = {}
        code_counter = 0

        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                code_counter += 1
                raw_content = part[3:-3]
                
                first_newline = raw_content.find('\n')
                lang = None
                code_content = raw_content
                
                if first_newline != -1:
                    first_line = raw_content[:first_newline].strip()
                    if first_line and first_line.isalnum() and len(first_line) < 20:
                        lang = first_line
                        code_content = raw_content[first_newline+1:]
                
                code_content = code_content.strip()
                
                # Store detected code
                ext = lang if lang else "txt"
                fname = f"snippet_{code_counter}.{ext}"
                detected_code_files[fname] = code_content

                highlighted_html = ""
                try:
                    if lang:
                        lexer = get_lexer_by_name(lang, stripall=True)
                    else:
                        lexer = guess_lexer(code_content)
                except:
                        # from pygments.lexers import PythonLexer
                        lexer = PythonLexer()
                    
                # PYGMENTS
                formatter = HtmlFormatter(style='default', noclasses=True, nowrap=True)
                highlighted_html = highlight(code_content, lexer, formatter)

                html_block = (
                    f"<table width='100%' bgcolor='#F8F8F8' border='0' cellspacing='0' cellpadding='10' style='margin-top:5px; margin-bottom:5px; border-radius:5px; border:1px solid #E0E0E0;'>"
                    f"<tr><td><pre style='font-family:Consolas,Monaco,monospace; font-size:16px; color:#333; margin:0; white-space:pre-wrap;'>"
                    f"{highlighted_html}</pre></td></tr></table>"
                )
                final_html_parts.append(html_block)
            else:
                if not part:
                    continue
                
                stripped_part = part.strip()
                if not stripped_part:
                    continue

                # Basic parsing: Remove bold/italic markers and convert headings to HTML
                # Remove ** ** for bold
                text_part = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', stripped_part)
                # Remove * * for italic
                text_part = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text_part)
                # Convert ## Heading to <b>Heading</b><br>
                text_part = re.sub(r'#{1,6}\s+(.*?)$', r'<b>\1</b><br>', text_part, flags=re.MULTILINE)
                
                safe_part = text_part # Skip html.escape to preserve our tags, but risky if user input has HTML. 
                # Better approach: Escape first, then apply regex.

                safe_part = html.escape(stripped_part)
                # Re-apply markdown-like formatting on escaped string
                # Bold
                safe_part = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_part)
                # Italic
                safe_part = re.sub(r'\*(.*?)\*', r'<i>\1</i>', safe_part)
                # Headings
                safe_part = re.sub(r'#{1,6}\s+(.*?)$', r'<b>\1</b>', safe_part, flags=re.MULTILINE)
                
                html_part = safe_part.replace("\n", "<br>")
                final_html_parts.append(f"<span>{html_part}</span>")
        
        final_html = "".join(final_html_parts)
        
        label = QLabel(f"<p style='line-height:120%; margin:0;'>{final_html}</p>")

        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setOpenExternalLinks(True)
        
        font_metrics = label.fontMetrics()
        
        # Width constraints
        if is_user:
             if len(text) > 10: 
                 label.setMinimumWidth(min(600, font_metrics.boundingRect(text).width() + 50))
             label.setMaximumWidth(1000) 
        else:
             # AI Message expanded width
             label.setMinimumWidth(200)
             # No maximum width or very large to allow expansion
        
        bubble_layout.addWidget(label)

        # Use explicitly provided files OR fallback to detected code blocks
        files_to_show = files if files else detected_code_files

        if files_to_show:
            btn_open_editor = QPushButton("Open in Code Editor")
            btn_open_editor.setCursor(Qt.PointingHandCursor)
            btn_open_editor.setStyleSheet("""
                QPushButton {
                    background-color: #28a745; 
                    color: white; 
                    border-radius: 5px; 
                    padding: 5px 10px; 
                    font-size: 12px; 
                    margin-top: 5px;
                }
                QPushButton:hover { background-color: #218838; }
            """)
            btn_open_editor.clicked.connect(lambda checked=False, f=files_to_show: self.open_in_editor(f))
            bubble_layout.addWidget(btn_open_editor)

        if is_user:
            bubble.setStyleSheet("QFrame { background-color: #95EC69; border-radius: 10px; border-top-right-radius: 2px; } QLabel { color: #000000; font-size: 18px; font-family: Arial;}")
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        else:
            # New UI for AI messages
            ai_container = QWidget()
            ai_layout = QVBoxLayout(ai_container)
            ai_layout.setContentsMargins(0, 0, 0, 0)
            ai_layout.setSpacing(5)
            
            # Header
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(5, 0, 0, 0)
            header_layout.setSpacing(8)
            
            avatar_label = QLabel()
            avatar_label.setFixedSize(24, 24)
            
            # Determine icon based on role
            role_icon_map = {
                "Supervisor": "Supervisor-Agent.png",
                "Developer": "Developer-Agent.png",
                "Support": "Support-Agent.png",
                "Scientist": "Scientist-Agent.png"
            }
            icon_filename = role_icon_map.get(display_role, "OpenMCS-Agent.png")
            logo_path = os.path.join(project_root, "resources", "logo", icon_filename)
            
            # Fallback to default if specific icon doesn't exist
            if not os.path.exists(logo_path):
                logo_path = os.path.join(project_root, "resources", "logo", "OpenMCS-Agent.png")

            if os.path.exists(logo_path):
                 pixmap = QPixmap(logo_path)
                 avatar_label.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                 avatar_label.setText("🤖")
                 avatar_label.setStyleSheet("background-color: #ddd; border-radius: 12px;")
            
            role_label = QLabel(display_role)
            role_label.setStyleSheet("font-weight: bold; color: #555; font-size: 14px;")
            
            header_layout.addWidget(avatar_label)
            header_layout.addWidget(role_label)
            header_layout.addStretch()
            
            ai_layout.addWidget(header_widget)
            
            # Bubble
            bubble.setStyleSheet("QFrame { background-color: #FFFFFF; border-radius: 10px; border-top-left-radius: 2px; border: none; } QLabel { color: #333333; font-size: 18px; font-family: Arial; }")
            ai_layout.addWidget(bubble)
            
            row_layout.addWidget(ai_container)
            # No addStretch() to allow full width

        self.messages_layout.addWidget(row_widget)
        self._scroll_to_bottom()

    def open_in_editor(self, files):
        """Open the provided files in the code editor as a new session."""
        if not self.code_editor:
            self.code_editor = CodeEditorWindow()
        
        if files:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            title = f"Chat Code {timestamp}"
            self.code_editor.add_session(title, files)
            
        self.code_editor.show()
        self.code_editor.raise_()

    def _scroll_to_bottom(self):
        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def on_send_clicked(self):
        text = self.input.toPlainText().strip()
        if not text: return

        self.add_message("user", text, is_user=True)
        self.input.clear()
        self.input.setDisabled(True)
        self.send_btn.setDisabled(True)

        self.file_paths = []
        self.display_files_text = ""
        self.label_files.clear()

        self.loading_dots = 0
        self.waiting_start_time = datetime.datetime.now()
        self.waiting_message_widget = self._add_waiting_message()
        self._update_loading_animation()
        self.loading_timer.start(1000)

        if not self.agent:
            self.agent = build_agent(self.cbox_model.currentText())

        self.worker = AgentWorker(self.agent, text, self.agent_config, self.agent_context)
        self.worker.result_ready.connect(self.handle_agent_response)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    @pyqtSlot(object)
    def handle_agent_response(self, response):
        self.loading_timer.stop()
        
        if hasattr(self, 'waiting_message_widget') and self.waiting_message_widget:
            self.messages_layout.removeWidget(self.waiting_message_widget)
            self.waiting_message_widget.deleteLater()
            self.waiting_message_widget = None
            self.waiting_message_label = None

        self.input.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.send_btn.setText("Send")
        self.input.setFocus()

        if isinstance(response, str) and response.startswith("Error:"):
            self.add_message("assistant", f"⚠️ {response}", is_user=False)
            return

        # Prefer structured_response if present; else render raw
        if isinstance(response, dict) and 'structured_response' in response:
            structured = response.get('structured_response')
            if hasattr(structured, 'assistant_message'):
                msg = structured.assistant_message
                self.add_message("assistant", msg, is_user=False, files=getattr(structured, 'files', None))
            else:
                # Fallback if structured exists but not in expected format
                self.add_message("assistant", str(structured), is_user=False)
        elif hasattr(response, 'assistant_message'):
            # ResponseFormat object directly
            msg = response.assistant_message
            self.add_message("assistant", msg, is_user=False, files=getattr(response, 'files', None))
        else:
            # Raw dict or string
            self.add_message("assistant", str(response), is_user=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OpenMCSChatWindow()
    w.show()
    app.processEvents()
    sys.exit(app.exec_())