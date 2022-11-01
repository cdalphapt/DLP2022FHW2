import torch
import torch.nn as nn
from torch.nn.modules import *
import torchvision.models as models

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size
        nn.Sigmoid(), 
        nn.MaxPool2d(2, 2), # kernel_size, stride
        nn.Conv2d(6, 16, 5),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
        nn.Linear(16*4*4, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 200)  #modified to 200
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


myResNet50 = models.__dict__['resnet50'](num_classes=200)
