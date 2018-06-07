import os
import sys
sys.path.insert(0, './')

import argparse
import numpy as np

import torch
import torch.nn as nn

from util.dataset import cifar10
from util.train import train_test
from util.lr_parser import lr_parser
from util.device_parser import parse_device_alloc

from cv.cifar10.models.cnn import ConvNet1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'input batch size, default = 128')
    parser.add_argument('--batch_size_test', type = int, default = None,
        help = 'batch size during test phrase, default is the same value during training')
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'the total number of epochs, default = 200')
    parser.add_argument('--lr_policy', type = str, default = 'exp_drop,0.1,0.1,100,150',
        help = 'learning rate policy, default = "exp_drop,0.1,0.1,100,150"')
    parser.add_argument('--momentum', type = float, default = 0.9,
        help = 'the momentum value, default = 0.9')
    parser.add_argument('--weight_decay', type = float, default = 1e-4,
        help = 'weight decay, default = 1e-4')

    parser.add_argument('--gpu', type = int, default = None,
        help = 'specify which gpu to use, default = None, supported values can be "cpu", "1", "1,2" etc.')
    parser.add_argument('--output_folder', type = str, default = None,
        help = 'the output folder')
    parser.add_argument('--model_name', type = str, default = 'model',
        help = 'the name of the model')

    args = parser.parse_args()
    args.batch_size_test = args.batch_size if args.batch_size_test == None else args.batch_size_test
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    train_loader, test_loader = cifar10(batch_size = args.batch_size, batch_size_test = args.batch_size_test)

    model = ConvNet1(input_size = [32, 32], input_channel = 3, output_class = 10, use_lrn = True)

    device, model = parse_device_alloc(device_config = args.gpu, model = model)

    lr_list = lr_parser(policy = args.lr_policy, epoch_num = args.epoch_num)

    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}
    setup_config['lr_list'] = lr_list
    optimizer = torch.optim.SGD(model.parameters(), lr_list[0], momentum = args.momentum, weight_decay = args.weight_decay)

    results = train_test(setup_config = setup_config, model = model, train_loader = train_loader, test_loader = test_loader, epoch_num = args.epoch_num,
        optimizer = optimizer, lr_list = lr_list, output_folder = args.output_folder, model_name = args.model_name, device = device)
