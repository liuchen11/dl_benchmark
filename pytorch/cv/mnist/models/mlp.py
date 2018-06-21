import torch
import torch.nn as nn

import numpy as np

class MLP(nn.Module):
    '''
    >>> General class for multilayer perceptron
    >>> Suitable for MNIST
    '''

    def __init__(self, input_dim = 784, hidden_dims = [], output_class = 10, dropout = None):
        '''
        >>> input_dim, hidden_dims, output_class: the dim of input neurons, hidden neurons and output neurons
        >>> dropout: the dropout rate i.e. the probability to deactivate the neuron, None means no dropout
        '''

        super(MLP, self).__init__()

        self.neurons = [input_dim,] + hidden_dims + [output_class,]

        self.layers = []

        dropout_each_layer = dropout
        if not isinstance(dropout_each_layer, (tuple, list)):
            dropout_each_layer = [dropout,] * len(hidden_dims)

        for in_dim, out_dim, dropout_this_layer in zip(self.neurons[:-2], self.neurons[1:-1], dropout_each_layer):
            sub_layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout_this_layer != None:
                sub_layers.append(nn.Dropout(p = dropout_this_layer))
            self.layers.append(nn.Sequential(*sub_layers))

        self.output = nn.Linear(self.neurons[-2], output_class)

    def forward(self, x):
        '''
        >>> x: 2-d tensor of shape [batch_size, in_dim]
        '''

        out = x.view(x.size(0), -1)
        for layer in self.layers:
            out = layer(out)
        out = self.output(out)

        return out
