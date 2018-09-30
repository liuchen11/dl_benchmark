import torch
import torch.nn as nn

class Basic_Generator(nn.Module):

    def __init__(self, width = 512, input_dim = 2, hidden_layers = 2, output_dim = 2):
        super(Basic_Generator, self).__init__()

        layer_list = [nn.Linear(input_dim, width), nn.ReLU(True)]
        for _ in range(hidden_layers):
            layer_list += [nn.Linear(width, width), nn.ReLU(True)]
        layer_list += [nn.Linear(width, output_dim)]

        self.layers = nn.Sequential(*layer_list)

    def forward(self, noise):
        output = self.layers(noise)
        return output

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

class Basic_Discriminator(nn.Module):

    def __init__(self, width = 512, input_dim = 2, hidden_layers = 2, output_dim = 1):
        super(Basic_Discriminator, self).__init__()

        layer_list = [nn.Linear(input_dim, width), nn.ReLU(True)]
        for _ in range(hidden_layers):
            layer_list += [nn.Linear(width, width), nn.ReLU(True)]
        layer_list += [nn.Linear(width, output_dim)]

        self.layers = nn.Sequential(*layer_list)

    def forward(self, inputs):
        output = self.layers(inputs)
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
