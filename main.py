import sys
import UI_MainWindow

from PyQt5.QtWidgets import QApplication
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UI_MainWindow.Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
