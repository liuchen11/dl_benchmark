import os
import sys
sys.path.insert(0, './')

import argparse
import numpy as np

import torch
import torch.nn as nn

from cv.mnist.models.mlp import MLP
from util.dataset import mnist
from util.train import train_test

from util.param_parser import DictParser, IntListParser
from util.lr_parser import parse_lr
from util.optim_parser import parse_optim
from util.device_parser import parse_device_alloc, config_visible_gpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 64,
        help = 'input batch size, default = 64')
    parser.add_argument('--batch_size_test', type = int, default = None,
        help = 'batch size during test phrase, default is the same value during training')
    parser.add_argument('--epoch_num', type = int, default = 50,
        help = 'the total number of epochs, default = 50')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [300, 100],
        help = 'neurons in each hidden layer, separated by ","')
    parser.add_argument('--dropout', type = float, default = None,
        help = 'dropout config, default = None')
    parser.add_argument('--lr_policy', action = DictParser,
        default = {'name': 'exp_decay', 'start_value': 0.01, 'decay_ratio': 0.95, 'decay_freq': 1},
        help = 'lr policy, default is name=exp_decay,start_value=0.01,decay_rato=0.95,decay_freq=1')
    parser.add_argument('--optim_policy', action = DictParser,
        default = {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
        help = 'optimizer config, default is name=sgd,lr=0.01,momentum=0.9,weight_decay=5e-4')

    parser.add_argument('--snapshots', action = IntListParser, default = None,
        help = 'check points to save some intermediate ckpts, default = None, values separated by ","')
    parser.add_argument('--gpu', type = str, default = None,
        help = 'specify which gpu to use, default = None')
    parser.add_argument('--output_folder', type = str, default = None,
        help = 'the output folder')
    parser.add_argument('--model_name', type = str, default = 'model',
        help = 'the name of the model')

    args = parser.parse_args()
    args.batch_size_test = args.batch_size if args.batch_size_test == None else args.batch_size_test
    config_visible_gpu(args.gpu)

    train_loader, test_loader = mnist(batch_size = args.batch_size, batch_size_test = args.batch_size_test)

    model = MLP(input_dim = 784, hidden_dims = args.hidden_dims, output_class = 10, dropout = args.dropout)

    device_ids, model = parse_device_alloc(device_config = None, model = model)

    lr_func = parse_lr(policy = args.lr_policy, epoch_num = args.epoch_num)
    optimizer = parse_optim(policy = args.optim_policy, params = model.parameters())

    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}
    setup_config['lr_list'] = [lr_func(idx) for idx in range(args.epoch_num)]
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    tricks = {}
    if args.snapshots != None:
        tricks['snapshots'] = args.snapshots

    results = train_test(setup_config = setup_config, model = model, train_loader = train_loader, test_loader = test_loader, epoch_num = args.epoch_num,
        optimizer = optimizer, lr_func = lr_func, output_folder = args.output_folder, model_name = args.model_name, device_ids = device_ids, **tricks)

