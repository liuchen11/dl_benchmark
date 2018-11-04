import os
import sys
sys.path.insert(0, './')

import argparse
import numpy as np

import torch
import torch.nn as nn

from cv.cifar10.models.densenet import DenseNet
from util.dataset import cifar10
from util.train import train_test
from util.param_parser import DictParser, IntListParser
from util.lr_parser import parse_lr
from util.optim_parser import parse_optim
from util.device_parser import parse_device_alloc, config_visible_gpu

from optim.ema import EMA

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 64,
        help = 'input batch size, default = 64')
    parser.add_argument('--batch_size_test', type = int, default = None,
        help = 'batch size during test phrase, default is the same value during training')

    parser.add_argument('--input_channels', type = int, default = 24,
        help = 'the number of input channels, default = 24')
    parser.add_argument('--growth_rate', type= int, default = 12,
        help = 'the grow rate of densenet, default = 12')
    parser.add_argument('--compression', type = float, default = 0.5,
        help = 'the compression rate of dense net, default = 0.5')
    parser.add_argument('--depth', type = int, default = 100,
        help = 'the number of total layers, default = 100')
    parser.add_argument('--efficient', type = int, default = 0,
        help = 'whether or not to use efficient implementation, default = 0, meaning no')

    parser.add_argument('--epoch_num', type = int, default = 300,
        help = 'the total number of epochs, default = 300')
    parser.add_argument('--lr_policy', action = DictParser,
        default = {'name': 'exp_drop', 'start_value': 0.1, 'decay_ratio': 0.1, 'milestones': '150_225'},
        help = 'lr policy, default is name=exp_drop,start_value=0.1,decay_ratio=0.1,milestones=150_225')
    parser.add_argument('--optim_policy', action = DictParser,
        default = {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4},
        help = 'optimizer config, default is name=sgd,lr=0.1,momentum=0.9,weight_decay=1e-4')
    parser.add_argument('--ema', type = float, default = None,
        help = 'the parameter for exponentially moving average, default = None, means no such trick')

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
    args.ema = None if args.ema < 0 else args.ema
    config_visible_gpu(args.gpu)

    train_loader, test_loader = cifar10(batch_size = args.batch_size, batch_size_test = args.batch_size_test)

    assert (args.depth - 4) % 6 == 0, 'the depth should be in the form like 6n+4'
    num_layer_each_block = (args.depth - 4) / 6
    block_config = [num_layer_each_block, num_layer_each_block, num_layer_each_block]
    model = DenseNet(input_channels = args.input_channels, growth_rate = args.growth_rate, block_config = block_config,
        compression = args.compression, efficient = False if args.efficient == 0 else True)

    device_ids, model = parse_device_alloc(device_config = None, model = model)

    lr_func = parse_lr(policy = args.lr_policy, epoch_num = args.epoch_num)
    optimizer = parse_optim(policy = args.optim_policy, params = model.parameters())

    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}
    setup_config['lr_list'] = [lr_func(idx) for idx in range(args.epoch_num)]
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    tricks = {}
    if args.ema != None:
        assert args.ema > 0 and args.ema < 1, 'The decaying ratio for EMA must be in (0, 1)'
        ema_wrapper = EMA(args.ema)
        ema_wrapper.register_model(model = model)
        tricks['ema'] = ema_wrapper
    if args.snapshots != None:
        tricks['snapshots'] = args.snapshots

    results = train_test(setup_config = setup_config, model = model, train_loader = train_loader, test_loader = test_loader, epoch_num = args.epoch_num,
        optimizer = optimizer, lr_func = lr_func, output_folder = args.output_folder, model_name = args.model_name, device_ids = device_ids, **tricks)


