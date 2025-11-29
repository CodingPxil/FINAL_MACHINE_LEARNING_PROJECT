import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class RES_MODEL(nn.Module):
    def __init__(self, in_d, h_d, num_classes):
        super().__init__()
        self.l1 = nn.Linear(in_d, h_d)


        self.r1 = nn.ReLU()
     

        self.l2 = nn.Linear(h_d, h_d//2)


        self.r2 = nn.ReLU()



        self.l3 = nn.Linear(h_d//2, num_classes)
    def forward(self, x):
        x = self.l1(x)

        x = self.r1(x)



        x = self.l2(x)

        x = self.r2(x)



        x= self.l3(x)
        return x




def build_model(num_classes, gray_scale, freeze_backbone, lr):
    my_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if(gray_scale==True):
        old_weight = my_model.conv1.weight.data
        new_conv = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)
        my_model.conv1 = new_conv


    features = my_model.fc.in_features
    my_model.fc = RES_MODEL(features, 256, num_classes)

    if freeze_backbone:
        for name, param in my_model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    train_params = []
    for param in my_model.parameters():
        if param.requires_grad == True:
            train_params.append(param)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr)


    return my_model, criterion, optimizer