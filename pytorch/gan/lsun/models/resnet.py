import sys
sys.path.insert(0, './')
from gan.util.models import BatchNorm2dLayer, LayerNormLayer

import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias = True):

        super(ConvLayer, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride = 1, padding = self.padding, bias = bias)

    def forward(self, input):

        output = self.conv(input)
        return output

class ResidualBlock(nn.Module):

    def __init__(self, input_dim, in_channel, out_channel, resample, kernel_size = 3):

        super(ResidualBlock, self).__init__()

        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2),
                ConvLayer(in_channel = in_channel, out_channel = out_channel, kernel_size = 1),
                )

            self.block = nn.Sequential(
                LayerNormLayer([in_channel, input_dim, input_dim]),
                nn.ReLU(),
                ConvLayer(in_channel = in_channel, out_channel = in_channel, kernel_size = kernel_size),
                LayerNormLayer([in_channel, input_dim, input_dim]),
                nn.ReLU(),
                ConvLayer(in_channel = in_channel, out_channel = out_channel, kernel_size = kernel_size),
                nn.AvgPool2d(kernel_size = 2),
                )
        elif resample == 'up':
            self.shortcut = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor = 2),
                ConvLayer(in_channel = in_channel, out_channel = out_channel, kernel_size = 1)
                )

            self.block = nn.Sequential(
                BatchNorm2dLayer(in_channel),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor = 2),
                ConvLayer(in_channel = in_channel, out_channel = out_channel, kernel_size = kernel_size),
                BatchNorm2dLayer(out_channel),
                nn.ReLU(),
                ConvLayer(in_channel = out_channel, out_channel = out_channel, kernel_size = kernel_size),
                )
        else:
            raise ValueError('invalid resample value')

    def forward(self, input):

        shortcut_part = self.shortcut(input)
        forward_part = self.block(input)

        return shortcut_part + forward_part

class Good_Discriminator(nn.Module):

    def __init__(self, input_dim = 64, in_channel = 3, num_channel = 64):

        print('num_channel = %s'%(num_channel))
        super(Good_Discriminator, self).__init__()

        self.input_dim = input_dim
        self.in_channel = in_channel
        self.num_channel = num_channel

        self.preprocess = ConvLayer(in_channel = in_channel, out_channel = num_channel, kernel_size = 3)

        self.resBlock = nn.Sequential(
            ResidualBlock(input_dim = 64, in_channel = num_channel, out_channel = 2 * num_channel, resample = 'down'),
            ResidualBlock(input_dim = 32, in_channel = 2 * num_channel, out_channel = 4 * num_channel, resample = 'down'),
            ResidualBlock(input_dim = 16, in_channel = 4 * num_channel, out_channel = 8 * num_channel, resample = 'down'),
            ResidualBlock(input_dim = 8, in_channel = 8 * num_channel, out_channel = 8 * num_channel, resample = 'down'),
            )

        self.outBlock = nn.Linear(4 * 4 * 8 * num_channel, 1)

    def forward(self, input):

        input = input.view(-1, self.in_channel, self.input_dim, self.input_dim)
        output = self.preprocess(input)
        output = self.resBlock(output)
        output = output.view(-1, 4 * 4 * 8 * self.num_channel)
        output = self.outBlock(output)

        return output.view(-1)

    def weight_init(self, init_type = 'normal', init_var = 0.02):

        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'normlayer' in name and 'weight' in name:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(1.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(1.0 - init_var, 1.0 + init_var)
                else:
                    raise ValueError('wrong init_type %s'%init_type)
            else:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(0.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(-init_var, init_var)
                else:
                    raise ValueError('wrong init_type %s'%init_type)

class Good_Generator(nn.Module):

    def __init__(self, input_dim = 128, num_channel = 64):

        print('channel = %s'%(num_channel))
        super(Good_Generator, self).__init__()

        self.num_channel = num_channel

        self.preprocess = nn.Linear(input_dim, 4 * 4 * 8 * num_channel)

        self.resBlock = nn.Sequential(
            ResidualBlock(input_dim = 4, in_channel = 8 * num_channel, out_channel = 8 * num_channel, resample = 'up'),
            ResidualBlock(input_dim = 8, in_channel = 8 * num_channel, out_channel = 4 * num_channel, resample = 'up'),
            ResidualBlock(input_dim = 16, in_channel = 4 * num_channel, out_channel = 2 * num_channel, resample = 'up'),
            ResidualBlock(input_dim = 32, in_channel = 2 * num_channel, out_channel = num_channel, resample = 'up'),
            )

        self.outBlock = nn.Sequential(
            BatchNorm2dLayer(num_channel),
            nn.ReLU(),
            ConvLayer(in_channel = num_channel, out_channel = 3, kernel_size = 3),
            nn.Tanh(),
            )

    def forward(self, input):

        output = self.preprocess(input)
        output = output.view(-1, 8 * self.num_channel, 4, 4)
        output = self.resBlock(output)
        output = self.outBlock(output)

        return output.view(output.shape[0], -1)

    def weight_init(self, init_type = 'normal', init_var = 0.02):

        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'normlayer' in name and 'weight' in name:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(1.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(1.0 - init_var, 1.0 + init_var)
                else:
                    raise ValueError('wrong init_type %s'%init_type)
            else:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(0.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(-init_var, init_var)
                else:
                    raise ValueError('wrong init_type %s'%init_type)
