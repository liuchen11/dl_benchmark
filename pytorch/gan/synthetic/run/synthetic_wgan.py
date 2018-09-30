import os
import sys
sys.path.insert(0, './')
import argparse
import numpy as np

import torch

from gan.util.train import wgan_train
from gan.util.plot_image import plot_synthetic_data
from gan.util.datasets import gaussian_8, gaussian_25, swiss_roll, gaussian_8_multivar
from gan.synthetic.models.basic import Basic_Generator, Basic_Discriminator
from util.device_parser import parse_device_alloc, config_visible_gpu
from util.param_parser import IntListParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'gaussian_25',
        help = 'Dataset, default = "gaussian_25", allowed = ["gaussian_25", "gaussian_8", "swiss_roll", "gaussian_8_multivar"]')
    parser.add_argument('--batch_size', type = int, default = 256,
        help = 'The batch size, default = 256')
    parser.add_argument('--gpu', type = str, default = "0",
        help = 'GPU devices, default = "0", meaning use GPU 0')
    parser.add_argument('--iter', type = int, default = 100000,
        help = 'The total iterations, default = 100000')
    parser.add_argument('--critic_iters', type = int, default = 5,
        help = 'The critic iteration number, default = 5')
    parser.add_argument('--grad_lambda', type = float, default = 0.1,
        help = 'Lambda value for the punishment term, default = 0.1')
    parser.add_argument('--output_folder', type = str, default = None,
        help = 'The output folder of the model')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')
    parser.add_argument('--save_ckpt', action = IntListParser, default = None,
        help = 'The list of saved ckpt, default = None')

    parser.add_argument('--print_freq', type = int, default = 1000,
        help = 'the frequency of printing the plots, default = 1000')
    parser.add_argument('--plot_samples', type = int, default = 5,
        help = 'the number of batches plotted, default = 5')

    args = parser.parse_args()
    config_visible_gpu(args.gpu)
    use_gpu = torch.cuda.is_available() and args.gpu not in ['cpu']

    if args.output_folder == None:
        raise ValueError('The output folder cannot be None')
    if args.model_name == None:
        raise ValueError('The name of the model cannot be None')
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    netG = Basic_Generator()
    netD = Basic_Discriminator()
    netG.weight_init()
    netD.weight_init()

    _, netG = parse_device_alloc(device_config = None, model = netG)
    _, netD = parse_device_alloc(device_config = None, model = netD)

    optimG = torch.optim.Adam(netG.parameters(), lr = 1e-4, betas = (0.5, 0.9))
    optimD = torch.optim.Adam(netD.parameters(), lr = 1e-4, betas = (0.5, 0.9))

    if args.dataset.lower() in ['gaussian_8']:
        data = gaussian_8(args.batch_size)
    elif args.dataset.lower() in ['gaussian_25']:
        data = gaussian_25(args.batch_size)
    elif args.dataset.lower() in ['swiss_roll']:
        data = swiss_roll(args.batch_size)
    elif args.dataset.lower() in ['gaussian_8_multivar']:
        data = gaussian_8_multivar(args.batch_size)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)

    plot_func = lambda output: plot_synthetic_data(batch_size = args.batch_size, batch_num = args.plot_samples,
            data = data, netG = netG, input_dim = 2, use_gpu = use_gpu, output_file = output)

    W_distance_list = wgan_train(netD = netD, netG = netG, optimD = optimD, optimG = optimG, iter_num = args.iter,
        critic_iters = args.critic_iters, data_loader = data, batch_size = args.batch_size, input_dim = 2,
        gp_lambda = args.grad_lambda, use_gpu = use_gpu, plot_freq = args.iter / 100, plot_func = plot_func,
        save_ckpt_list = args.save_ckpt, model_name = args.model_name, output_folder = args.output_folder)

    
