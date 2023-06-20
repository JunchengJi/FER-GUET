import cv2
import os

import setting

# 加载人脸检测器模型
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

# 输入和输出目录路径
input_dir = '../dataset/temp'
output_dir = '../dataset/temp_gray'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有图片文件
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 加载图片并将其转换为灰度图像
        image = cv2.imread(os.path.join(input_dir, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测图像中的人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # 在图像中绘制矩形框标记出检测到的人脸，并保存矩形框内的人脸图像
        for i, (x, y, w, h) in enumerate(faces):
            # 绘制矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 剪切出人脸图像并保存
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(output_dir, f'faces_{filename}'), face_img)

        # 保存带有人脸矩形框的图像
        # cv2.imwrite(os.path.join(output_dir, f'detected_faces_{filename}'), image)
