# DLP2022FHW2

A Simple test project for Dropout within ResNet.

To run this program, make sure you have installed all packages mentioned in https://github.com/pytorch/examples/blob/main/imagenet/main.py , and torchinfo is also required for this program.

The usage is similar with https://github.com/pytorch/examples/blob/main/imagenet/main.py , but here are several changes:

1. Run main.py to start training.
2. --arch parameter is no longer in use.
3. Use --myarch id to change the model to be trained. id is an interger between 0 and 4.
		 0: original resnet18
		 1: resnet18 with dropout
		 2: resnet18 with smaller initial conv
		 3: arch 2 with additional fullconnection layer
		 4: arch 3 with dropout
   The invalid id will lead to the program training original resnet18.
   

