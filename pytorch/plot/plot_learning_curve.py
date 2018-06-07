import os
import sys
sys.path.insert(0, './')
if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle
import argparse
import numpy as np

from plot.color import get_color

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_file', type = str, default = None,
        help = 'the file(s) to be loaded')
    parser.add_argument('--metric', type = str, default = None,
        help = 'to plot the loss or error rate, default = None')
    parser.add_argument('--output', type = str, default = None,
        help = 'the output image, default = None')

    args = parser.parse_args()
    args.pkl_file = args.pkl_file.split(',')

    if args.output != None:
        if os.sep in args.output and not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    if args.metric == None:
        loss_ax = ax1
        err_ax = ax1.twinx()
    elif args.metric.lower() in ['loss']:
        loss_ax = ax1
        err_ax = None
    elif args.metric.lower() in ['err']:
        loss_ax = None
        err_ax = ax1
    else:
        raise ValueError('Unrecognized metric %s'%args.metric)

    color_idx = 0
    handle_list = []
    ax1.set_xlabel('Epoch')
    if args.metric == None or args.metric.lower() in ['loss']:
        loss_ax.set_ylabel('Loss')
        loss_metric_list = ['train_loss', 'validate_loss', 'test_loss']
        for pkl_file in args.pkl_file:
            info = pickle.load(open(pkl_file, 'rb'))
            name = pkl_file.split(os.sep)[-1]
            for loss_metric in loss_metric_list:
                if loss_metric in info:
                    loss_info = info[loss_metric]
                    epoch_list = list(sorted(loss_info.keys()))
                    value_list = [loss_info[idx] for idx in epoch_list]
                    handle, = loss_ax.plot(epoch_list, value_list, label = name + ' ' + loss_metric, color = get_color(color_idx))
                    handle_list.append(handle)
                    color_idx += 1
    
    if args.metric == None or args.metric.lower() in ['err']:
        err_ax.set_ylabel('Error')
        err_metric_list = ['train_acc', 'validate_acc', 'test_acc']
        for pkl_file in args.pkl_file:
            info = pickle.load(open(pkl_file, 'rb'))
            name = pkl_file.split(os.sep)[-1]
            for err_metric in err_metric_list:
                if err_metric in info:
                    err_info = info[err_metric]
                    epoch_list = list(sorted(err_info.keys()))
                    value_list = [1. - err_info[idx] for idx in epoch_list]
                    handle, = err_ax.plot(epoch_list, value_list, label = name + ' ' + err_metric, color = get_color(color_idx))
                    handle_list.append(handle)
                    color_idx += 1

    label_list = [h.get_label() for h in handle_list]
    ax1.legend(handle_list, label_list)

    if args.output == None:
        plt.show()
    else:
        plt.savefig(args.output, dpi = 500, bbox_inches = 'tight')

