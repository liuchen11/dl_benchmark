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
    >>> Same structure as TensorFlow tutorial
    >>> Suitable for MNIST
    '''

    def __init__(self, input_size = [28, 28], input_channels = 1, output_class = 10):

        super(ConvNet1, self).__init__()
        image_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        image_size = [int(image_size[0] // 2), int(image_size[1] // 2)]

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        image_size = [int(image_size[0] // 2), int(image_size[1] // 2)]

        fc_in_dim = np.prod(image_size) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.ReLU()
            )

        self.output = nn.Linear(512, output_class)

    def forward(self, x):
        '''
        >>> x: 4-d tensor of shape [batch_size, channel_num, height, width]
        '''

        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.output(out)

        return out
