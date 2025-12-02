
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class RES_MODEL(nn.Module):
    def __init__(self, in_d, h_d, num_classes):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_d, h_d),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(h_d, h_d//2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(h_d//2, num_classes)
        )
       
    def forward(self, x):
      
        return self.layer(x)




def build_model(num_classes, gray_scale, freeze_backbone, lr,h_d=256):
    # Load Pre-trained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if gray_scale:
        old_weight = model.conv1.weight.data.clone()
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

    # .pth file expects a single Linear layer here
    in_features = model.fc.in_features
    model.fc = RES_MODEL(in_features, h_d, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        train_params = model.fc.parameters()
    else:
        train_params = model.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr)

    return model, criterion, optimizer