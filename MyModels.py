import torch
import torch.nn as nn
from torch.nn.modules import *
import torchvision.models as models
from collections import OrderedDict

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

def GenerateMyModel():
    myResNet50 = models.__dict__['resnet50'](num_classes = 200)
    initConv = nn.Sequential(
        OrderedDict(
            [
               ('smallerConv', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False))
            ])
        )    
    myResNet50.conv1 = initConv
    #classifier = nn.Sequential(
    #    OrderedDict(
    #        [('fc1', nn.Linear(2048, 1024)),
    #        ('relu1', nn.ReLU()), 
    #        #('dropout1',nn.Dropout(0.5)),
    #        ('fc2', nn.Linear(1024, 512)),
    #        ('relu1', nn.ReLU()), 
    #        #('dropout1',nn.Dropout(0.5)),
    #        ('output', nn.Linear(512, 200)),
    #        ])
    #    )
    #myResNet50.fc = classifier
    return myResNet50


