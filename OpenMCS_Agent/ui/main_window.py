import os
import sys
import re
import html 
import uuid
import datetime
import tempfile
import shutil

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
# from core.agent import build_agent # Removed heavy import
# from core.multi_agent import build_multi_agent_graph # Removed heavy import
from core.schemas import Context
from ui.widgets import ChatInput
from ui.worker import AgentWorker, AgentInitializeWorker
from ui.code_editor import CodeEditorWindow

# from utils.document_loader import load_html, load_pdf, load_json # Moved to usage to avoid heavy import

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
        self.current_attachments = [] # List of temp file paths for images

        self._init_ui()
        
        self.add_message("assistant", HELLO_MESSAGE, is_user=False)
        
        # Initialize agent asynchronously
        self.init_agent(self.cbox_model.currentText())

    def init_agent(self, provider):
        """Start background initialization of the agent."""
        self.enable_input(False)
        self.add_message("system", f"Initializing agent with **{provider}**... Please wait.", is_user=False)
        
        self.init_worker = AgentInitializeWorker(provider)
        self.init_worker.finished_signal.connect(self.on_agent_initialized)
        self.init_worker.error_signal.connect(self.on_agent_init_error)
        self.init_worker.start()

    def on_agent_initialized(self, agent, provider):
        self.agent = agent
        self.enable_input(True)
        self.add_message("system", f"Agent initialized successfully with **{provider}**.", is_user=False)
        # Remove the initializing message or just leave the success message
        
    def on_agent_init_error(self, error_msg):
        self.add_message("system", f"❌ Failed to initialize agent: {error_msg}", is_user=False)
        self.enable_input(True) # Re-enable so user can try again or change model

    def enable_input(self, enable: bool):
        self.input.setEnabled(enable)
        self.send_btn.setEnabled(enable)
        self.cbox_model.setEnabled(enable)

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
        input_wrapper = QWidget()
        input_wrapper_layout = QVBoxLayout(input_wrapper)
        input_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        input_wrapper_layout.setSpacing(5)

        # Image Attachments Preview
        self.attachments_widget = QWidget()
        self.attachments_layout = QHBoxLayout(self.attachments_widget)
        self.attachments_layout.setAlignment(Qt.AlignLeft)
        self.attachments_layout.setContentsMargins(5, 5, 5, 5)
        self.attachments_widget.hide()
        input_wrapper_layout.addWidget(self.attachments_widget)

        # Chat Input Row
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(10)

        self.input = ChatInput()
        self.input.sendMessage.connect(self.on_send_clicked)
        self.input.pasteImage.connect(self.on_image_pasted)
        self.input.fileDropped.connect(self.on_file_added)

        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedHeight(60)
        self.send_btn.setFixedWidth(150)
        self.send_btn.clicked.connect(self.on_send_clicked)

        input_layout.addWidget(self.input)
        input_layout.addWidget(self.send_btn)
        
        input_wrapper_layout.addWidget(input_container)
        main_layout.addWidget(input_wrapper, 0)

    def on_model_changed(self):
        """切换模型时重新构建 Agent"""
        provider = self.cbox_model.currentText()
        # Use asynchronous initialization to prevent freezing
        self.init_agent(provider)

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
                    # Lazy import to avoid startup freeze
                    from utils.document_loader import load_html, load_pdf

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

    def on_image_pasted(self, image):
        # Save QImage to temp file
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(temp_file.name, "PNG")
            self.current_attachments.append(temp_file.name)
            
            # Update UI
            self.refresh_attachments_preview()
        except Exception as e:
            print(f"Error saving pasted image: {e}")

    def on_file_added(self, file_path):
        """Handle dropped or pasted file paths"""
        if file_path not in self.current_attachments:
            self.current_attachments.append(file_path)
            self.refresh_attachments_preview()

    def refresh_attachments_preview(self):
        # Clear layout
        while self.attachments_layout.count():
            item = self.attachments_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.current_attachments:
            self.attachments_widget.hide()
            return

        self.attachments_widget.show()
        for idx, file_path in enumerate(self.current_attachments):
            container = QWidget()
            container.setFixedSize(90, 90) # Fixed container size for absolute positioning
            
            # Thumbnail (bottom-left aligned in the container)
            lbl = QLabel(container)
            pixmap = QPixmap(file_path).scaled(80, 80, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pixmap)
            lbl.setFixedSize(80, 80)
            lbl.move(0, 10) # Offset to make room for top button
            lbl.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
            
            # Delete button (top-right overlapping)
            btn_del = QPushButton("×", container)
            btn_del.setFixedSize(24, 24)
            btn_del.move(68, 0) # Overlap the top-right corner of the image (which is at x=0+80=80, y=10)
            
            btn_del.setCursor(Qt.PointingHandCursor)
            btn_del.clicked.connect(lambda checked, i=idx: self.remove_attachment_at(i))
            btn_del.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444; 
                    color: white; 
                    border-radius: 12px; 
                    font-weight: bold; 
                    font-size: 16px;
                    padding: 0px;
                    border: 2px solid white;
                }
                QPushButton:hover { background-color: #cc0000; }
            """)
            
            self.attachments_layout.addWidget(container)

    def remove_attachment_at(self, index):
        if 0 <= index < len(self.current_attachments):
            path = self.current_attachments.pop(index)
            # Optional: delete file? Keep for history message maybe?
            # If we delete now, we can't send it. If we send it, we shouldn't delete immediately if async?
            # We copy for history, so it's fine.
            self.refresh_attachments_preview()

    def clear_attachments(self):
        # We don't delete files here immediately as they might be used in history
        self.current_attachments.clear()
        self.refresh_attachments_preview()

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

    def add_message(self, role, text, is_user: bool, files: dict = None, images: list = None):
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
        bubble_layout.setSpacing(5) # Increased spacing
        
        # 1. Display Images if any
        if images:
             for img_path in images:
                 try:
                     lbl = QLabel()
                     pixmap = QPixmap(img_path)
                     if not pixmap.isNull():
                         # Scale down if too big
                         if pixmap.width() > 500:
                             pixmap = pixmap.scaledToWidth(500, Qt.SmoothTransformation)
                         lbl.setPixmap(pixmap)
                         bubble_layout.addWidget(lbl)
                 except Exception as e:
                     print(f"Error displaying image in history: {e}")

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
        attachments = list(self.current_attachments)
        
        if not text and not attachments: return

        self.add_message("user", text, is_user=True, images=attachments)
        self.input.clear()
        self.clear_attachments()
        
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

        self.worker = AgentWorker(self.agent, text, self.agent_config, self.agent_context, images=attachments)
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