# Exponentially Moving Average

import torch

class EMA(object):

    def __init__(self, mu):

        self.mu = mu
        self.shadow = {}
        print('Constructing a EMA wrapper with mu = %1.1e'%self.mu)

    def register(self, name, value):

        self.shadow[name] = value.clone()

    def update(self, name, value):

        assert name in self.shadow
        average = self.mu * value + (1. - self.mu) * self.shadow[name]
        self.shadow[name] = average.clone()
        return average

    def register_model(self, model):

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param)

    def update_model(self, model, overwrite = True):

        for name, param in model.named_parameters():
            if param.requires_grad:
                if overwrite:
                    param.data = self.update(name, param.data)
                else:
                    self.update(name, param.data)
