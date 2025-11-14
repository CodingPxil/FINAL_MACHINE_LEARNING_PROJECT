import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(num_classes, gray_scale, freeze_backbone, lr):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if(gray_scale==True):
        model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)

   
    if freeze_backbone:
        train_params = model.fc.parameters() 
    else:
        train_params = model.parameters()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr)


    return model, criterion, optimizer