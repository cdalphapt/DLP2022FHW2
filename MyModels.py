import torch
import torch.nn as nn
from torch.nn.modules import *
import torchvision.models as models
from collections import OrderedDict

def GenerateMyModel(mode):
    #0:vanilla resnet18
    #1:resnet18 with dropout
    #2:resnet18 with smaller initial conv
    #3:resnet18 with smaller initial conv and additional full connection
    #4:resnet18 with smaller initial conv and additional full connection plus dropout
    #default: mode 0
    mode = int(mode)

    myResNet18 = models.__dict__['resnet18'](num_classes = 200)

    if mode == 2 or mode == 3 or mode == 4:
        initConv = nn.Sequential(
            OrderedDict(
                [
                   ('smallerConv', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False))
                ])
            )    
        myResNet18.conv1 = initConv
    if mode == 3:
        classifier = nn.Sequential(
            OrderedDict(
                [('fc1', nn.Linear(512, 384)),
                ('relu1', nn.ReLU()), 
                ('fc2', nn.Linear(384, 256)),
                ('relu2', nn.ReLU()), 
                ('output', nn.Linear(256, 200)),
                ])
            )
        myResNet18.fc = classifier
    if mode == 4:
        classifier = nn.Sequential(
            OrderedDict(
                [('fc1', nn.Linear(512, 384)),
                ('relu1', nn.ReLU()), 
                ('dropout1',nn.Dropout(0.5)),
                ('fc2', nn.Linear(384, 256)),
                ('relu2', nn.ReLU()), 
                ('dropout2',nn.Dropout(0.5)),
                ('output', nn.Linear(256, 200)),
                ])
            )
        myResNet18.fc = classifier
    if mode == 1:
        singleDO = nn.Sequential(
            OrderedDict(
                [('fc1', nn.Linear(512, 200)),
                ('dropout1',nn.Dropout(0.5)),
                ])
            )
        myResNet18.fc = singleDO
    return myResNet18


