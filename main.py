import sys
from PyQt5.QtWidgets import QApplication
from OpenMCS_Agent.ui.main_window import OpenMCSChatWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OpenMCSChatWindow()
    w.show()
    app.processEvents()
    sys.exit(app.exec_())