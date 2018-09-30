import torch
import torch.nn as nn

class MNIST_Generator(nn.Module):

    def __init__(self, input_dim = 128, hidden_dim = 64, output_dim = 784):
        super(MNIST_Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 4 * self.hidden_dim),
            nn.ReLU(True),
            )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 5),
            nn.ReLU(True),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 5),
            nn.ReLU(True),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 1, 8, stride = 2)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise):

        output = self.preprocess(noise).view(-1, 4 * self.hidden_dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.block3(output)
        output = self.sigmoid(output)

        return output.view(-1, self.output_dim)

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

class MNIST_Discriminator(nn.Module):

    def __init__(self, input_dim = 28, hidden_dim = 64):

        super(MNIST_Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, 5, stride = 2, padding = 2),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, 5, stride = 2, padding = 2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.hidden_dim, 4 * self.hidden_dim, 5, stride = 2, padding = 2),
            nn.ReLU(True),
            )
        self.block2 = nn.Sequential(
            nn.Linear(4 * 4 * 4 * self.hidden_dim, 1)
            )

    def forward(self, input):

        output = input.view(-1, 1, self.input_dim, self.input_dim)
        output = self.block1(output)
        output = output.view(-1, 4 * 4 * 4 * self.hidden_dim)
        output = self.block2(output)
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
