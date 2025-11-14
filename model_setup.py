import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(num_classes, lr):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    featutres= model.fc.in_features
    model.fc = nn.Linear(featutres, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    return model, criterion, optimizer