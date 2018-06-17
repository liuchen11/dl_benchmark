import torch
import torch.nn as nn

import numpy as np

class ConvNet(nn.Module):

    def __init__(self,):
        super(ConvNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Should not call an abstract function')

class ConvNet1(ConvNet):
    '''
    >>> ConvNet consisting of two convolutional layers and two fully connected layers
    >>> Same structure for TensorFlow tutorial
    >>> Suitable for MNIST
    '''

    
