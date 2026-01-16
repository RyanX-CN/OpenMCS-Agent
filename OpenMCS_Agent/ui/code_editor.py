import sys
import os
import json
import uuid
import datetime
import subprocess
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPlainTextEdit, 
    QPushButton, QLabel, QSplitter, QComboBox, QTextEdit, QMessageBox, QMenu, QAction, QInputDialog, QToolBar,
    QToolButton
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat

# Path to save sessions
SESSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code_sessions_history.json")

class PythonHighlighter(QSyntaxHighlighter):
    """Simple Python Syntax Highlighter"""
    def __init__(self, document):
        super().__init__(document)
        self.highlightingRules = []

        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(QColor("#569CD6")) # Blue-ish
        keywordFormat.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "False", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "None",
            "nonlocal", "not", "or", "pass", "raise", "return", "True",
            "try", "while", "with", "yield"
        ]
        for word in keywords:
            pattern = f"\\b{word}\\b"
            self.highlightingRules.append((pattern, keywordFormat))

        # Strings
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(QColor("#CE9178")) # Orange-ish
        self.highlightingRules.append(("\"[^\"]*\"", stringFormat))
        self.highlightingRules.append(("'[^']*'", stringFormat))

        # Comments
        commentFormat = QTextCharFormat()
        commentFormat.setForeground(QColor("#6A9955")) # Green-ish
        self.highlightingRules.append(("#[^\n]*", commentFormat))

    def highlightBlock(self, text):
        import re
        for pattern, format in self.highlightingRules:
            expression = re.compile(pattern)
            for match in expression.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)

class CodeEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Editor & Runner")
        self.resize(1000, 700)
        
        # Data structure: { session_id: { "name": str, "files": { filename: content } } }
        self.sessions = {} 
        self.current_session_id = None
        self.current_file = None
        self.is_dark_theme = False

        self._init_ui()
        self._apply_theme()
        self._load_sessions()

    def _init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- MenuBar ---
        menubar = self.menuBar()
        settings_action = menubar.addAction("Settings")
        settings_menu = QMenu()
        settings_action.setMenu(settings_menu)

        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(theme_action)

        # --- Toolbar ---
        # toolbar = QToolBar("Main Toolbar")
        # toolbar.setMovable(False)
        # self.addToolBar(toolbar)

        # Splitter for Sidebar vs Editor
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        sidebar_label = QLabel("Code Sessions")
        sidebar_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(sidebar_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        sidebar_layout.addWidget(self.tree)
        
        sidebar_widget.setLayout(sidebar_layout)
        splitter.addWidget(sidebar_widget)

        # --- Editor Area ---
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar (Editor specific)
        editor_toolbar_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Run Code")
        self.run_btn.clicked.connect(self.run_code)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        editor_toolbar_layout.addWidget(self.run_btn)
        
        editor_toolbar_layout.addStretch()
        editor_layout.addLayout(editor_toolbar_layout)

        # Code Editor
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.editor.textChanged.connect(self.on_text_changed)
        editor_layout.addWidget(self.editor, 2) # Stretch factor 2

        # Output Area
        output_label = QLabel("Output:")
        editor_layout.addWidget(output_label)
        
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 10))
        self.output_area.setFixedHeight(150)
        editor_layout.addWidget(self.output_area, 1) # Stretch factor 1

        editor_widget.setLayout(editor_layout)
        splitter.addWidget(editor_widget)

        # Set initial splitter sizes
        splitter.setSizes([250, 750])
        main_layout.addWidget(splitter)

    def add_session(self, title, files):
        """Add a new session with multiple files. Merges if session with same title exists."""
        # Check for existing session by title
        session_id = None
        for sid, data in self.sessions.items():
            if data["name"] == title:
                session_id = sid
                break
        
        root = None
        if session_id:
            # Find existing root item
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if item.data(0, Qt.UserRole) == session_id:
                    root = item
                    break
        else:
            # Create new session
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {"name": title, "files": {}}
            
            root = QTreeWidgetItem(self.tree)
            root.setText(0, title)
            root.setData(0, Qt.UserRole, session_id)
        
        # Add files
        first_added_item = None
        
        for fname, content in files.items():
            # Handle cases where content is a dict (LLM structured output issue)
            if isinstance(content, dict):
                content = content.get('code', content.get('content', content.get('text', str(content))))
            
            # Ensure strictly string
            if not isinstance(content, str):
                content = str(content)

            final_fname = fname
            # Ensure unique filename if file already exists
            if final_fname in self.sessions[session_id]["files"]:
                base, ext = os.path.splitext(fname)
                counter = 1
                while final_fname in self.sessions[session_id]["files"]:
                    final_fname = f"{base}_{counter}{ext}"
                    counter += 1
            
            self.sessions[session_id]["files"][final_fname] = content
            
            child = QTreeWidgetItem(root)
            child.setText(0, final_fname)
            child.setData(0, Qt.UserRole, session_id)
            child.setData(0, Qt.UserRole + 1, final_fname)
            
            if first_added_item is None:
                first_added_item = child
            
        self.tree.expandItem(root)
        self._save_sessions()
        
        # Select the first added file
        if first_added_item:
            self.tree.setCurrentItem(first_added_item)
            self.on_item_clicked(first_added_item, 0)

    def on_item_clicked(self, item, column):
        session_id = item.data(0, Qt.UserRole)
        filename = item.data(0, Qt.UserRole + 1)
        
        if not session_id or session_id not in self.sessions:
            return

        if filename:
            # It's a file
            content = self.sessions[session_id]["files"].get(filename, "")
            self.current_session_id = session_id
            self.current_file = filename
            self.editor.setPlainText(content)
            self.run_btn.setEnabled(True)
            self.editor.setEnabled(True)
        else:
            # It's a root (session)
            self.current_session_id = session_id
            self.current_file = None
            self.editor.clear()
            self.editor.setPlaceholderText("Select a file to edit...")
            self.run_btn.setEnabled(False)
            self.editor.setEnabled(False)

    def on_text_changed(self):
        """Save changes to memory as user types."""
        if self.current_session_id and self.current_file:
            content = self.editor.toPlainText()
            self.sessions[self.current_session_id]["files"][self.current_file] = content
            # We don't save to disk on every keystroke, maybe on close or specific actions
            # But for safety, let's save to disk periodically or on focus lost? 
            # For now, let's save on run or switch. 
            # Actually, let's just save to disk here but maybe throttle it? 
            # Simplest: save on text change is fine for local JSON if not too huge.
            # To avoid lag, we can skip saving to disk here and only save memory.
            pass 

    def run_code(self):
        if not self.current_session_id or not self.current_file:
            return

        # Ensure current content is saved
        content = self.editor.toPlainText()
        self.sessions[self.current_session_id]["files"][self.current_file] = content
        self._save_sessions() # Save to disk before running
        
        self.output_area.clear()
        self.output_area.append(f"Running {self.current_file}...\n")

        # Run in subprocess
        process = QProcess(self)
        process.readyReadStandardOutput.connect(lambda: self.handle_output(process))
        process.readyReadStandardError.connect(lambda: self.handle_error(process))
        process.finished.connect(lambda: self.output_area.append("\n[Finished]"))
        
        # Use the same python interpreter
        process.start(sys.executable, ["-c", content])

    def handle_output(self, process):
        data = process.readAllStandardOutput().data().decode()
        self.output_area.insertPlainText(data)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def handle_error(self, process):
        data = process.readAllStandardError().data().decode()
        self.output_area.insertPlainText(data)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def open_context_menu(self, position):
        item = self.tree.itemAt(position)
        menu = QMenu()
        
        if not item:
            # Clicked on empty space
            new_session_action = QAction("New Session", self)
            new_session_action.triggered.connect(self.create_new_session)
            menu.addAction(new_session_action)
        else:
            session_id = item.data(0, Qt.UserRole)
            filename = item.data(0, Qt.UserRole + 1)
            
            if not filename: # It's a session root
                new_file_action = QAction("New File", self)
                new_file_action.triggered.connect(lambda: self.create_new_file(session_id))
                menu.addAction(new_file_action)
                
                menu.addSeparator()
                
                rename_action = QAction("Rename Session", self)
                rename_action.triggered.connect(lambda: self.rename_session(item, session_id))
                menu.addAction(rename_action)
                
                delete_action = QAction("Delete Session", self)
                delete_action.triggered.connect(lambda: self.delete_session(item, session_id))
                menu.addAction(delete_action)
            else:
                # It's a file
                rename_file_action = QAction("Rename File", self)
                rename_file_action.triggered.connect(lambda: self.rename_file(session_id, filename))
                menu.addAction(rename_file_action)

                delete_file_action = QAction("Delete File", self)
                delete_file_action.triggered.connect(lambda: self.delete_file(session_id, filename, item))
                menu.addAction(delete_file_action)
        
        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def create_new_session(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        title = f"New Session {timestamp}"
        self.add_session(title, {"main.py": ""})

    def create_new_file(self, session_id):
        new_filename, ok = QInputDialog.getText(self, "New File", "Filename:", text="new_script.py")
        if ok and new_filename:
            if new_filename in self.sessions[session_id]["files"]:
                QMessageBox.warning(self, "Error", "File already exists!")
                return
            
            self.sessions[session_id]["files"][new_filename] = ""
            self._save_sessions()
            
            # Find root item
            root = None
            for i in range(self.tree.topLevelItemCount()):
                it = self.tree.topLevelItem(i)
                if it.data(0, Qt.UserRole) == session_id:
                    root = it
                    break
            
            if root:
                child = QTreeWidgetItem(root)
                child.setText(0, new_filename)
                child.setData(0, Qt.UserRole, session_id)
                child.setData(0, Qt.UserRole + 1, new_filename)
                root.setExpanded(True)
                self.tree.setCurrentItem(child)
                self.on_item_clicked(child, 0)

    def rename_file(self, session_id, old_filename):
        new_filename, ok = QInputDialog.getText(self, "Rename File", "New Filename:", text=old_filename)
        if ok and new_filename and new_filename != old_filename:
            if new_filename in self.sessions[session_id]["files"]:
                QMessageBox.warning(self, "Error", "File already exists!")
                return
            
            content = self.sessions[session_id]["files"].pop(old_filename)
            self.sessions[session_id]["files"][new_filename] = content
            self._save_sessions()
            self._load_sessions()

    def delete_file(self, session_id, filename, item=None):
        confirm = QMessageBox.question(self, "Delete File", 
                                     f"Are you sure you want to delete {filename}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            del self.sessions[session_id]["files"][filename]
            self._save_sessions()
            
            if item:
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
            else:
                self._load_sessions()
            
            if self.current_file == filename:
                self.editor.clear()
                self.current_file = None

    def rename_session(self, item, session_id):
        old_name = self.sessions[session_id]["name"]
        new_name, ok = QInputDialog.getText(self, "Rename Session", "New Name:", text=old_name)
        if ok and new_name:
            self.sessions[session_id]["name"] = new_name
            item.setText(0, new_name)
            self._save_sessions()

    def delete_session(self, item, session_id):
        confirm = QMessageBox.question(self, "Delete Session", 
                                     "Are you sure you want to delete this session and all its files?",
                                     QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            del self.sessions[session_id]
            # Remove from tree
            index = self.tree.indexOfTopLevelItem(item)
            self.tree.takeTopLevelItem(index)
            
            if self.current_session_id == session_id:
                self.editor.clear()
                self.current_session_id = None
                self.current_file = None
                
            self._save_sessions()

    def _save_sessions(self):
        try:
            with open(SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")

    def _load_sessions(self):
        if not os.path.exists(SESSION_FILE):
            return
            
        try:
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                self.sessions = json.load(f)
                
            self.tree.clear()
            for session_id, data in self.sessions.items():
                root = QTreeWidgetItem(self.tree)
                root.setText(0, data["name"])
                root.setData(0, Qt.UserRole, session_id)
                
                for fname in data["files"].keys():
                    child = QTreeWidgetItem(root)
                    child.setText(0, fname)
                    child.setData(0, Qt.UserRole, session_id)
                    child.setData(0, Qt.UserRole + 1, fname)
        except Exception as e:
            print(f"Error loading sessions: {e}")

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self._apply_theme()

    def _apply_theme(self):
        if self.is_dark_theme:
            # Dark Theme
            self.setStyleSheet("""
                QWidget { background-color: #1E1E1E; color: #D4D4D4; }
                QTreeWidget { background-color: #252526; border: 1px solid #3E3E42; }
                QTreeWidget::item:selected { background-color: #37373D; }
                QPlainTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #3E3E42; }
                QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #3E3E42; }
                QPushButton { background-color: #3E3E42; color: white; border: none; padding: 5px 10px; }
                QPushButton:hover { background-color: #505050; }
                QSplitter::handle { background-color: #3E3E42; }
                QMenu { background-color: #252526; color: #D4D4D4; border: 1px solid #3E3E42; }
                QMenu::item:selected { background-color: #37373D; }
            """)
        else:
            # Light Theme
            self.setStyleSheet("""
                QWidget { background-color: #F0F0F0; color: #000000; }
                QTreeWidget { background-color: #FFFFFF; border: 1px solid #CCCCCC; }
                QTreeWidget::item:selected { background-color: #E0E0E0; color: black; }
                QPlainTextEdit { background-color: #FFFFFF; color: #000000; border: 1px solid #CCCCCC; }
                QTextEdit { background-color: #F8F8F8; color: #000000; border: 1px solid #CCCCCC; }
                QPushButton { background-color: #E0E0E0; color: black; border: 1px solid #CCCCCC; padding: 5px 10px; }
                QPushButton:hover { background-color: #D0D0D0; }
                QSplitter::handle { background-color: #CCCCCC; }
                QMenu { background-color: #FFFFFF; color: #000000; border: 1px solid #CCCCCC; }
                QMenu::item:selected { background-color: #E0E0E0; }
            """)
