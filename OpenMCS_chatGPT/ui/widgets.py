from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTextOption

class ChatInput(QTextEdit):
    sendMessage = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setPlaceholderText("Please enter a message... (\"Enter\" to send, \"Shift+Enter\" for newline)")
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setFixedHeight(60)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.sendMessage.emit()
        else:
            super().keyPressEvent(event)