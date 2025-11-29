import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(num_classes, gray_scale, freeze_backbone, lr):
    # Load Pre-trained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if gray_scale:
        old_weight = model.conv1.weight.data
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

    # .pth file expects a single Linear layer here
    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    if freeze_backbone:
        train_params = model.fc.parameters()
    else:
        train_params = model.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr)

    return model, criterion, optimizer