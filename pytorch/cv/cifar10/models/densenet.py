import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import math
import numpy as np

class BNFactory(object):
    '''
    >>> Batch normalization factory class
    '''

    def __init__(self, norm, relu, conv):
        '''
        >>> norm: normalization layer
        >>> relu: relu function
        >>> conv: convolutional layer
        '''
        self.norm = norm
        self.relu = relu
        self.conv = conv

    def __call__(self, *inputs):
        '''
        >>> *inputs: list of feature maps, shape of [batch_size, channels, height, width]
        '''
        concated_features = torch.cat(inputs, 1)
        bottlenect_output = self.conv(self.relu(self.norm(concated_features)))
        return bottlenect_output

class DenseLayer(nn.Module):

    def __init__(self, input_channels, growth_rate, bn_size, drop_rate, efficient = False):
        '''
        >>> input_channels: total number of input channels
        >>> growth_rate: the number of output channels
        >>> bn_size: batch norm size
        >>> drop_rate: the dropout probability
        >>> efficient: whether or not to use efficient implementation, default = False
        '''
        super(DenseLayer, self).__init__()
        self.main_block = nn.Sequential()

        self.main_block.add_module('norm1', nn.BatchNorm2d(input_channels))
        self.main_block.add_module('relu1', nn.ReLU(inplace = True))
        self.main_block.add_module('conv1', nn.Conv2d(input_channels, bn_size * growth_rate, kernel_size = 1, stride = 1, bias = False))

        self.main_block.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.main_block.add_module('relu2', nn.ReLU(inplace = True))
        self.main_block.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False))

        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        '''
        >>> *prev_features: the list of all previous layers
        '''
        bn_layer = BNFactory(norm = self.main_block.norm1, relu = self.main_block.relu1, conv = self.main_block.conv1)
        if self.efficient and any(prev_features.requires_grad for prev_feature in prev_features):
            bottlenect_output = cp.checkpoint(bn_layer, *prev_features)
        else:
            bottlenect_output = bn_layer(*prev_features)

        out_features = self.main_block.conv2(self.main_block.relu2(self.main_block.norm2(bottlenect_output)))
        if self.drop_rate > 0:
            out_features = F.dropout(out_features, p = self.drop_rate, training = self.training)
        return out_features

class Transition(nn.Module):

    def __init__(self, input_channels, output_channels):
        '''
        >>> input_channels: number of input channels
        >>> output_channels: number of output channels

        >>> convolutional layer which functions as a transition
        >>> the height and width of each feature maps halved after this transformation
        '''
        super(Transition, self).__init__()
        self.main_block = nn.Sequential()
        self.main_block.add_module('norm', nn.BatchNorm2d(input_channels))
        self.main_block.add_module('relu', nn.ReLU(inplace = True))
        self.main_block.add_module('conv', nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = 1, bias = False))
        self.main_block.add_module('pool', nn.AvgPool2d(kernel_size = 2, stride = 2))

    def forward(self, input):

        return self.main_block(input)

class DenseBlock(nn.Module):

    def __init__(self, num_layers, input_channels, bn_size, growth_rate, drop_rate, efficient = False):
        '''
        >>> num_layers: the number of layers
        >>> input_channels: total number of input channels
        >>> bn_size: batch normalization size
        >>> growth_rate: the number of additional channels in each dense layer
        >>> drop_rate: the dropout probability
        >>> efficient: whether or not to use efficient implementation, default = False
        '''
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                input_channels + i * growth_rate,
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate,
                efficient = efficient,
                )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, input):
        features = [input,]
        for name, layer in self.named_children():
            out_features = layer(*features)
            features.append(out_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):

    def __init__(self, input_channels = 24, bn_size = 4, growth_rate = 12, drop_rate = 0,
        block_config = (16, 16, 16), compression = 0.5, num_classes = 10, efficient = False):
        '''
        >>> input_channels: the output channels of first convolutional block i.e. the input channels to the main-body densely connected layers
        >>> bn_size: batch normalization size
        >>> growth_rate: the number of additional channels in each dense layer
        >>> drop_rate: the dropout probability
        >>> block_config: list/tuple of int, number of layers in each densely connected block
        >>> compression: compression rate
        >>> num_classes: number of output categories
        >>> efficient: whether or not to use efficient implementation
        '''
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8

        self.main_block = nn.Sequential()
        # First convolution
        self.main_block.add_module('conv0', nn.Conv2d(3, input_channels, kernel_size = 3, stride = 1, padding = 1, bias = False))

        # Each denseblock
        channel_num = input_channels
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers = num_layers,
                input_channels = channel_num,
                bn_size = bn_size,
                growth_rate = growth_rate,
                drop_rate = drop_rate,
                efficient = efficient,
                )
            self.main_block.add_module('denseblock%d' % (i + 1), block)
            channel_num = channel_num + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(input_channels = channel_num, output_channels = int(channel_num * compression))
                self.main_block.add_module('transition%d' % (i + 1), trans)
                channel_num = int(channel_num * compression)

        # Final Batch Norm
        self.main_block.add_module('norm_final', nn.BatchNorm2d(channel_num))

        # Linear Layer
        self.classifier = nn.Linear(channel_num, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input):

        output = self.main_block(input)
        output = F.relu(output, inplace = True)
        output = F.avg_pool2d(output, kernel_size = self.avgpool_size).view(output.size(0), -1)
        output = self.classifier(output)
        return output
