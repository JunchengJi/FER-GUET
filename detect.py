import os

import cv2
import torch
import torchvision
from torch import nn
from torchvision import models, transforms, datasets
from PIL import Image
import torch.nn.functional as F

import setting

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(setting.detection_model_path)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载表情resnet模型
model_path = setting.emotion_model
emotion_model = models.resnet18()
emotion_model.fc = nn.Linear(emotion_model.fc.in_features, 7)
emotion_model.load_state_dict(torch.load(model_path))
emotion_model.eval()

# 人脸识别相关
data_dir = 'dataset/face_dataset'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
class_names = image_datasets['train'].classes
class_sum = len(class_names)
# 加载模型
model = torchvision.models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_sum)
model.load_state_dict(torch.load('face_recognition/resnet18_face_recognition_self.pth'))


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
    outputs = emotion_model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def detect_face(image_path):
    # 转换图像
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image_path)
    image_tensor = data_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加 batch dimension

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = F.softmax(outputs, dim=1)
        print('outputs:', outputs)
        score, predicted = torch.max(outputs, 1)
        print("score: ", score)
        print("predicted: ", predicted)

    if score.double() > 1.0/class_sum:
        predicted_class = class_names[predicted[0]]
    else:
        predicted_class = 'UnKnown'
    print(predicted_class)
    return predicted_class
