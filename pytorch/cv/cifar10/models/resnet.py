import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DownSample1(nn.Module):
    '''
    >>> 1st downsample layer, decrease height and width by 2 and double the channels by padding
    '''

    def __init__(self,):
        super(DownSample1, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size = 1, stride = 2)    # Subsampling

    def forward(self, x):

        out = self.avgpool(x)
        channel_num = out.size(1)
        return F.pad(out, (0, 0, 0, 0, (channel_num + 1) // 2, channel_num // 2), 'constant', 0)

class ResBlock(nn.Module):
    '''
    >>> The building blocks of deep residual network, containing two conv layer and potentially one down sample layer
    '''

    def __init__(self, input_channels, output_channels, stride = 1, downsample = None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = output_channels, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(output_channels)
            )

        self.downsample = downsample

    def forward(self, x):

        base = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample != None:
            base = self.downsample(x)
        out = nn.ReLU()(out + base)

        return out

class ResNet(nn.Module):

    def __init__(self,):
        super(ResNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Should not call an abstract method')

    def _make_block(self, input_channels, output_channels, mini_blocks, stride = 1):

        assert stride in [1, 2], 'the stride here can be either 1 or 2'

        downsample = None if stride == 1 else DownSample1()

        layer_list = [ResBlock(input_channels = input_channels, output_channels = output_channels, stride = stride, downsample = downsample)]
        for layer_idx in range(1, mini_blocks):
            layer_list.append(ResBlock(input_channels = output_channels, output_channels = output_channels))

        return nn.Sequential(*layer_list)

class ResNet_3b(ResNet):

    def __init__(self, input_size = [32, 32], input_channels = 3, depth = 44, output_class = 10):

        super(ResNet_3b, self).__init__()

        assert (depth - 2) % 6 == 0, 'The depth of ResNet_3b must have the shape of 6x + 2'
        mini_blocks = (depth - 2) // 6

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )

        self.block1 = self._make_block(input_channels = 16, output_channels = 16, mini_blocks = mini_blocks, stride = 1)
        self.block2 = self._make_block(input_channels = 16, output_channels = 32, mini_blocks = mini_blocks, stride = 2)
        self.block3 = self._make_block(input_channels = 32, output_channels = 64, mini_blocks = mini_blocks, stride = 2)

        self.avgpool = nn.AvgPool2d(kernel_size = (input_size[0] // 2 // 2, input_size[1] // 2 // 2))

        self.classifier = nn.Linear(64, output_class)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        '''
        >>> x: 4-d tensor of shape [batch_size, channel_num, height, width]
        '''

        out = self.layer1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

