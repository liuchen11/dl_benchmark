import sys
sys.path.insert(0, './')
from gan.util.models import BatchNorm2dLayer

import torch
import torch.nn as nn

class DC_Generator(nn.Module):

    def __init__(self, input_dim = 128, num_channel = 64):

        super(LSUN_Generator, self).__init__()

        print('DC_Generator: num_channel = %s, add_norm = %s'%(num_channel, add_norm))

        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 16 * num_channel, kernel_size = 4, stride = 1, padding = 0),

            nn.ConvTranspose2d(16 * num_channel, 8 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(8 * num_channel),
            nn.ReLU(),

            nn.ConvTranspose2d(8 * num_channel, 4 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(4 * num_channel),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * num_channel, 2 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(2 * num_channel),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * num_channel, 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(),
            )

    def forward(self, input):

        output = input.view(input.shape[0], -1, 1, 1)
        output = self.block(output)
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

class DC_Discriminator(nn.Module):

    def __init__(self, input_dim = 64, in_channel = 3, num_channel = 64, add_norm = False):

        super(LSUN_Discriminator, self).__init__()

        self.input_dim = input_dim
        self.in_channel = in_channel

        print('DC_Discriminator: num_channel = %s, add_norm = %s'%(num_channel, add_norm))

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, 2 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),

            nn.Conv2d(2 * num_channel, 4 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(4 * num_channel),
            nn.ReLU(),

            nn.Conv2d(4 * num_channel, 8 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(8 * num_channel),
            nn.ReLU(),

            nn.Conv2d(8 * num_channel, 16 * num_channel, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2dLayer(16 * num_channel),
            nn.ReLU(),

            nn.Conv2d(16 * num_channel, 1, kernel_size = 4, stride = 1, padding = 0),
            )

    def forward(self, input):

        output = input.view(input.shape[0], self.in_channel, self.input_dim, self.input_dim)
        output = self.block(output)
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
