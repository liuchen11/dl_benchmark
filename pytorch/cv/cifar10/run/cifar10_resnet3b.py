import os
import sys
sys.path.insert(0, './')

import argparse
import numpy as np

import torch
import torch.nn as nn

from cv.cifar10.models.resnet import ResNet_3b
from util.dataset import cifar10
from util.train import train_test
from util.param_parser import DictParser
from util.lr_parser import parse_lr
from util.optim_parser import parse_optim
from util.device_parser import parse_device_alloc, config_visible_gpu

from optim.ema import EMA

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'input batch size, default = 128')
    parser.add_argument('--batch_size_test', type = int, default = None,
        help = 'batch size during test phrase, default is the same value during training')
    parser.add_argument('--depth', type = int, default = 44,
        help = 'the depth of the resnet, default = 44')
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'the total number of epochs, default = 200')
    parser.add_argument('--lr_policy', action = DictParser,
        default = {'name': 'exp_drop', 'start_value': 0.1, 'decay_ratio': 0.1, 'milestones': '100_150'},
        help = 'lr policy, default is name=exp_drop,start_value=0.1,decay_rato=0.1,milestones=100_150')
    parser.add_argument('--optim_policy', action = DictParser,
        default = {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4},
        help = 'optimizer config, default is name=sgd,lr=0.1,momentum=0.9,weight_decay=1e-4')
    parser.add_argument('--ema', type = float, default = None,
        help = 'the parameter for exponentially moving average, default = None, means no such trick')

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

    model = ResNet_3b(input_size = [32, 32], input_channels = 3, depth = args.depth, output_class = 10)

    device_ids, model = parse_device_alloc(device_config = None, model = model)

    lr_list = parse_lr(policy = args.lr_policy, epoch_num = args.epoch_num)
    optimizer = parse_optim(policy = args.optim_policy, params = model.parameters())

    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}
    setup_config['lr_list'] = lr_list
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    tricks = {}
    if args.ema != None:
        assert args.ema > 0 and args.ema < 1, 'The decaying ratio for EMA must be in (0, 1)'
        ema_wrapper = EMA(args.ema)
        ema_wrapper.register_model(model = model)
        tricks['ema'] = ema_wrapper

    results = train_test(setup_config = setup_config, model = model, train_loader = train_loader, test_loader = test_loader, epoch_num = args.epoch_num,
        optimizer = optimizer, lr_list = lr_list,  output_folder = args.output_folder, model_name = args.model_name, device_ids = device_ids, **tricks)
