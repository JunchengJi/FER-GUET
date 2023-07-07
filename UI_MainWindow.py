# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'UI_MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import csv
import datetime
import os
from statistics import mode
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QBrush, QColor
from PyQt5.QtWidgets import QFileDialog

import detect
import setting
from VideoThread import VideoThread


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setting_dialog = None
        self.video_thread = None
        self.setting_icon = None
        self.stop_icon = None
        self.start_icon = None
        self.video_default_bg = None
        self.btn_select_image = None
        self.btn_stop = None
        self.btn_start = None
        self.horizontalLayout = None
        self.video_label = None
        self.verticalLayout = None
        self.verticalLayoutWidget = None
        self.centralwidget = None
        self.setupUi()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1000, 800)
        self.setMaximumSize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setMinimumSize(QtCore.QSize(1000, 800))
        self.centralwidget.setMaximumSize(QtCore.QSize(1000, 800))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 980, 780))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        # 视频流
        self.video_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("video_label")
        self.verticalLayout.addWidget(self.video_label)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        # 开始按键
        self.btn_start = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_start.setObjectName("btn_start")
        self.horizontalLayout.addWidget(self.btn_start)
        # 停止按键
        self.btn_stop = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_stop.setObjectName("btn_stop")
        self.horizontalLayout.addWidget(self.btn_stop)
        # 图片选择按键
        self.btn_select_image = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_select_image.setObjectName("btn_setting")
        self.horizontalLayout.addWidget(self.btn_select_image)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowIcon(QIcon(setting.main_icon))
        self.setWindowTitle(_translate("MainWindow", "FER GUET"))
        self.video_label.setStyleSheet("border: 1px solid #ccc; border-radius: 10px;")
        self.video_label.setScaledContents(True)  # 图片拉伸填充
        image = QImage(setting.background_image)
        rounded_img = rounded_image(image, 10)
        self.video_default_bg = QPixmap.fromImage(rounded_img)
        self.video_label.setPixmap(self.video_default_bg)

        self.start_icon = QIcon(setting.start_icon)
        self.btn_start.setIcon(self.start_icon)
        self.btn_start.setIconSize(self.btn_start.size())

        self.stop_icon = QIcon(setting.stop_icon)
        self.btn_stop.setIcon(self.stop_icon)
        self.btn_stop.setIconSize(self.btn_stop.size())
        self.btn_stop.setEnabled(False)

        self.setting_icon = QIcon(setting.select_image_button)
        self.btn_select_image.setIcon(self.setting_icon)
        self.btn_select_image.setIconSize(self.btn_select_image.size())

        # 点击信号绑定
        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_select_image.clicked.connect(self.select_file)

    def start_video(self):
        self.btn_start.setEnabled(False)
        self.btn_select_image.setEnabled(False)
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_video_label)
        self.video_thread.start()
        self.btn_stop.setEnabled(True)

    def stop_video(self):
        self.btn_stop.setEnabled(False)
        self.video_thread.stop()
        self.video_thread = None
        # Load the image from file
        # self.image = QPixmap('resource/resource.png')
        # self.video_label.setPixmap(self.image)
        self.btn_start.setEnabled(True)
        self.btn_select_image.setEnabled(True)

    def select_file(self):
        image = QFileDialog.getOpenFileName(self, 'Open file', 'D:/test_image', 'Image files (*.jpg *.png)')
        if image[0]:
            # 读取图像
            img = cv2.imread(image[0])
            # 将图像转换为NumPy数组
            img_array = np.array(img)
            self.update_video_label(img_array)

    def showMain(self, mainWindow):
        self.show()
        mainWindow.close()

    def update_video_label(self, frame):
        frame = frame.copy()
        # Convert the frame to RGB format and create a Qt image from it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 获得图片 frame中所有的人脸位置
        faces = detect.face_detection.detectMultiScale(gray, 1.3, 5)
        # 对于所有发现的人脸
        emotion = None
        get_img_time = None
        name = None
        for (x, y, w, h) in faces:
            get_img_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

            # 获取人脸图像
            face = frame[y:y + h, x:x + w]

            emotion_arg = detect.resnet_detect_emotion(face)
            emotion = setting.emotion_labels[int(emotion_arg)]
            setting.emotion_window.append(emotion)

            if len(setting.emotion_window) >= setting.frame_window:
                setting.emotion_window.pop(0)

            # 获得出现次数最多的分类
            emotion_mode = mode(setting.emotion_window)
            name = detect.detect_face(face)
            text_label = name + '__' + emotion_mode
            # 在矩形框上部，输出分类文字
            cv2.putText(frame, text_label, (x, y - 30), setting.font, .7, (255, 140, 0), 1, cv2.LINE_AA)

        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        rounded_img = rounded_image(image, 10)
        # Create a pixmap from the Qt image and set it as the video label's pixmap
        pixmap = QPixmap.fromImage(rounded_img)
        self.video_label.setPixmap(pixmap)
        # 输出日志
        if not os.path.exists(setting.emotion_data_path):
            os.mkdir(setting.emotion_data_path)
            if emotion:
                write_to_csv(emotion, get_img_time, name)
        else:
            if emotion:
                write_to_csv(emotion, get_img_time, name)


def write_to_csv(emotion, time, name):
    with open(setting.emotion_data_path + datetime.datetime.now().strftime('%Y-%m-%d') + '.csv', mode='a',
              newline='') as csvfile:
        fieldnames = ['emotion', 'time', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 如果文件为空，写入表头
        if csvfile.tell() == 0:
            writer.writeheader()
        # 写入当前表情和时间
        writer.writerow({'emotion': emotion, 'time': time, 'name': name})


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
