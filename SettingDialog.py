from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap, QPainter, QBrush, QColor
from PyQt5.QtWidgets import QDialog, QPushButton, QHBoxLayout, QVBoxLayout

import setting


class SettingDialog(QDialog):
    closed = QtCore.pyqtSignal(QtWidgets.QMainWindow)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Setting")
        self.setWindowIcon(QIcon(setting.main_icon))
        self.resize(600, 520)
        font = QFont()
        font.setPointSize(14)
        self.setFont(font)

        self.image_label = QtWidgets.QLabel('')
        self.image_label.setScaledContents(True)  # 图片拉伸填充
        image = QImage(setting.background_image)
        rounded_img = rounded_image(image, 10)
        bg = QPixmap.fromImage(rounded_img)
        self.image_label.setPixmap(bg.scaled(360, 400))
        self.image_label.setAlignment(Qt.AlignCenter)
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.image_label)

        self.label_model = QtWidgets.QLabel('模型路径')
        self.lineEdit_model_path = QtWidgets.QLineEdit()
        self.pbtn_select_model = QPushButton('打开')
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.label_model)
        model_layout.addWidget(self.lineEdit_model_path)
        model_layout.addWidget(self.pbtn_select_model)

        self.label_output = QtWidgets.QLabel('日志路径')
        self.lineEdit_output_path = QtWidgets.QLineEdit()
        self.pbtn_select_output = QPushButton('打开')
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.label_output)
        output_layout.addWidget(self.lineEdit_output_path)
        output_layout.addWidget(self.pbtn_select_output)

        # Create the start and stop buttons
        self.pbtn_save = QPushButton('保存', self)
        self.pbtn_cancel = QPushButton('取消', self)
        # Create the layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pbtn_save)
        button_layout.addWidget(self.pbtn_cancel)

        # Create the layout for the main window
        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(button_layout)

        self.pbtn_save.clicked.connect(self.save)
        self.pbtn_cancel.clicked.connect(self.cancel)
        # Set the main window's layout
        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.closed.emit(self)
        event.accept()

    def save(self):
        self.accept()
        self.closed.emit(self)

    def cancel(self):
        self.reject()
        self.closed.emit(self)


def rounded_image(image, radius):
    # 创建带有圆角的 QImage
    rounded = QImage(image.size(), QImage.Format_ARGB32_Premultiplied)
    rounded.fill(Qt.transparent)

    # 创建 QPainter 和 QBrush 对象
    painter = QPainter(rounded)
    brush = QBrush(QColor(0, 0, 0, 255))
    painter.setBrush(brush)

    # 绘制圆角矩形
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawRoundedRect(image.rect(), radius, radius)

    # 将圆角矩形用作蒙版并裁剪 QImage
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.drawImage(0, 0, image)

    # 清理
    painter.end()

    return rounded
