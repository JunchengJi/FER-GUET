import cv2
import os
import numpy as np

# 定义数据目录和人物列表
data_dir = 'dataset/face_dataset/'
subjects = os.listdir(data_dir)

# 将每个人的图像和标签添加到列表中，并进行数据预处理
training_data = []
labels = []
for i, subject in enumerate(subjects):
    subject_dir = os.path.join(data_dir, subject)
    for image_name in os.listdir(subject_dir):
        image_path = os.path.join(subject_dir, image_name)
        image = cv2.imread(image_path)

        # 对图像进行直方图均衡化和对比度增强
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # 调整图像大小
        resized_image = cv2.resize(image, (100, 100))
        training_data.append(resized_image)
        labels.append(i)

# 提取人脸特征并训练模型
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(training_data, np.array(labels))

# 保存模型
recognizer.save('trainer.yml')
