import os

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import models, transforms, datasets
from PIL import Image

import setting

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
# 加载模型
model = torchvision.models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
# model.softmax = nn.Softmax(dim=1)
model.load_state_dict(torch.load('face_recognition/resnet18_face_recognition_self.pth'))
model.eval()
print(class_names)
print(model)
print(1 / len(class_names))
