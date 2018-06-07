import torch
import torch.nn as nn

import numpy as np

class ConvNet(nn.Module):

    def __init__(self,):
        super(ConvNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Should not call an abstract method')

class ConvNet1(ConvNet):
    '''
    >>> ConvNet consisting two convolutional layer and three fully connected layer
    >>> Suitable for CIFAR10
    '''

    def __init__(self, input_size = [32, 32], input_channel = 3, output_class = 10, use_lrn = True):

        super(ConvNet1, self).__init__()
        image_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            )
        self.lrn1 = nn.LocalResponseNorm(8, alpha = 0.001 / 9.0, beta = 0.75) if use_lrn else None
        image_size = [int(image_size[0] / 2), int(image_size[1] / 2)]

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            )
        self.lrn2 = nn.LocalResponseNorm(8, alpha = 0.001 / 9.0, beta = 0.75) if use_lrn else None
        image_size = [int(image_size[0] / 2), int(image_size[1] / 2)]

        fc_in_dim = np.prod(image_size) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(fc_in_dim, 384),
            nn.ReLU()
            )

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU()
            )

        self.output = nn.Linear(192, output_class)

    def forward(self, x):
        '''
        >>> x: 4-d tensor of shape [batch_size, channel_num, height, width]
        '''

        out = self.conv1(x)
        out = self.lrn1(out) if self.lrn1 != None else out
        out = self.conv2(out)
        out = self.lrn2(out) if self.lrn2 != None else out
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)

        return out

