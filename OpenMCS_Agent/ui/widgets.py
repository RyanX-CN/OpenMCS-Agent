from PyQt5.QtWidgets import QTextEdit, QApplication
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTextOption, QImage, QKeySequence

class ChatInput(QTextEdit):
    sendMessage = pyqtSignal()
    pasteImage = pyqtSignal(QImage)
    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setPlaceholderText("Please enter a message... (\"Enter\" to send, \"Shift+Enter\" for newline)")
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setFixedHeight(60)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setAcceptDrops(True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.sendMessage.emit()
        elif event.matches(QKeySequence.Paste):
            mime = QApplication.clipboard().mimeData()
            if mime.hasImage():
                event.accept()
                self.pasteImage.emit(QApplication.clipboard().image())
            elif mime.hasUrls():
                event.accept()
                for url in mime.urls():
                    file_path = url.toLocalFile()
                    if self._is_image(file_path):
                        self.fileDropped.emit(file_path)
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self._is_image(file_path):
                    self.fileDropped.emit(file_path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _is_image(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.png','.tif', '.tiff', '.webp'))