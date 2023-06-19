import os

import cv2
import numpy as np
import torch

import setting

data_dir = 'dataset/face_dataset/'
subjects = os.listdir(data_dir)
# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(setting.detection_model_path)

# 加载表情识别模型
emotion_classifier = torch.load(setting.classification_model_path)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载训练好的模型
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer.yml')


def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images / 255.0
    return images


def detect_emotion(face):
    try:
        # shape变为(48,48)
        face = cv2.resize(face, (48, 48))
    except:
        pass
    # 扩充维度，shape变为(1,48,48,1)
    # 将（1，48，48，1）转换成为(1,1,48,48)
    face = np.expand_dims(face, 0)
    face = np.expand_dims(face, 0)

    # 人脸数据归一化，将像素值从0-255映射到0-1之间
    face = preprocess_input(face)
    new_face = torch.from_numpy(face)
    new_new_face = new_face.float().requires_grad_(False)

    # 调用我们训练好的表情识别模型，预测分类
    emotion_arg = np.argmax(emotion_classifier.forward(new_new_face.to(DEVICE)).detach().cpu().numpy())
    return emotion_arg


def detect_face(face):
    # 检测人脸
    label, confidence = face_recognizer.predict(face)
    # 根据置信度判断是否识别成功
    if confidence < 120:
        name = subjects[label]
    else:
        name = "Unknown"
    return name, str(round(confidence, 2))
