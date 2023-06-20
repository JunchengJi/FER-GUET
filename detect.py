import os

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

import setting

data_dir = 'dataset/face_dataset/'
subjects = os.listdir(data_dir)
# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(setting.detection_model_path)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载人脸识别的模型
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer.yml')

# 加载resnet模型
model_path = setting.emotion_model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(model_path))
model.eval()


def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images / 255.0
    return images


def resnet_detect_emotion(image_path):
    # 加载图片并预处理
    image = Image.fromarray(image_path)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    # 进行预测
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def detect_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    label, confidence = face_recognizer.predict(gray)
    # 根据置信度判断是否识别成功
    if confidence < 100:
        name = subjects[label]
    else:
        name = "Unknown"
    return name, str(round(confidence, 2))
