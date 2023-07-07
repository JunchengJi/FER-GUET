import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 定义数据变换
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
device = torch.device("cpu")

# 加载数据集
image_datasets = {x: datasets.ImageFolder('dataset/' + x, data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}


def train_resNet(model, criterion, optimizer, scheduler, epochs):
    best_model = None
    best_acc = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                with torch.set_grad_enabled(phase == 'train'):  # 仅在train阶段启用梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # 如果当前阶段是“train”，则使用loss.backward（）计算损失相对于模型参数的梯度，并使用optimizer.step（）更新参数。
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds == labels.data).int())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{}: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_model = model
                best_acc = epoch_acc
    return best_model


# 保存模型
if __name__ == '__main__':
    # 定义模型
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 7)
    resnet.to(device)

    # 定义损失函数和优化器
    cri = nn.CrossEntropyLoss()
    opt = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    # 定义学习率调整策略
    exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    # 训练模型
    num_epochs = 20
    resNet_model = train_resNet(resnet, cri, opt, exp_lr_scheduler, num_epochs)
    # 保存模型
    torch.save(resNet_model.state_dict(), "models/resnet_model_cpu.pt")
