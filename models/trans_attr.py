# A CNN to predict transient attributes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TA_ResNet(nn.Module):        # ResNet-based model
    def __init__(self):
        super(TA_ResNet, self).__init__()

        self.base_net = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])  

        self.last_block = nn.Sequential(*list(models.resnet50(pretrained=True).children())[-3:-1])

        # freeze the base network
        for param in self.base_net.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 40)


    def forward(self, img):
        y = self.base_net(img)

        y = self.last_block(y)

        y = self.fc1(y.squeeze(3).squeeze(2))
        y = F.leaky_relu(y, 0.1)

        y = self.fc2(y)

        y = torch.sigmoid(y)

        return y
