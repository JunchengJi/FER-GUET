import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.video_capture = cv2.VideoCapture(0)  # 采样 fps = 30
        self.signal_interval = 3  # 信号传输的间隔帧数,此时理论实际用于预测 fps = 30/signal_interval
        self.cnt = 0  # 帧计数器

    def run(self):
        while self._run_flag:
            ret, frame = self.video_capture.read()
            if ret:
                self.cnt += 1
                if self.cnt % self.signal_interval == 0:
                    frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
                    self.change_pixmap_signal.emit(frame)
                    self.cnt = 0

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
