import sys
from PyQt5.QtWidgets import QApplication
from source.OpenMCS_chatGPT.ui.main_window import OpenMCSChatWindow
# from source.OpenMCS_codeGPT.ui.main_window import OpenMCSCodeWindow 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OpenMCSChatWindow()
    w.show()
    app.processEvents()
    sys.exit(app.exec_())