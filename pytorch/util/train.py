import os
import sys
sys.path.insert(0, './')
if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

from util.evaluation import AverageCalculator
from util.evaluation import accuracy

def train_test(setup_config, model, train_loader, test_loader, epoch_num,
    optimizer, lr_list, output_folder, model_name, device_ids,
    criterion = nn.CrossEntropyLoss(), **tricks):
    '''
    >>> general training function without validation set
    '''

    tosave = {'model_summary': str(model), 'setup_config': setup_config, 'train_loss': {}, 'train_acc': {}, 'test_loss': {}, 'test_acc': {}}
    device = torch.device('cuda:0' if not device_ids in ['cpu'] and torch.cuda.is_available() else 'cpu')
    if not device_ids in ['cpu']:
        criterion = criterion.cuda(device)
        model = model.cuda(device)

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()
    for epoch_idx in range(epoch_num):

        lr_this_epoch = lr_list[epoch_idx]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epoch

        print('Epoch %d: lr = %1.2e'%(epoch_idx, lr_this_epoch))

        acc_calculator.reset()
        loss_calculator.reset()
        model.train()                 # Switch to Train Mode
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            if not device_ids in ['cpu']: # Use of GPU
                data_batch_var = Variable(data_batch).cuda(device)
                label_batch = label_batch.cuda(device, async = True)
                label_batch_var = Variable(label_batch)
            else:
                data_batch_var = Variable(data_batch)
                label_batch_var = Variable(label_batch)

            logits = model(data_batch_var)
            loss = criterion(logits, label_batch_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))

            if 'ema' in tricks and tricks['ema'] != None:
                tricks['ema'].update_model(model = model)

        print('Training loss after epoch %d: %.4f'%(epoch_idx + 1, loss_calculator.average))
        tosave['train_acc'][epoch_idx] = acc_calculator.average
        tosave['train_loss'][epoch_idx] = loss_calculator.average

        acc_calculator.reset()
        loss_calculator.reset()
        model.eval()                    # Switch to Evaluation Mode
        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            if not device_ids in ['cpu']:   # Use of GPU
                data_batch = Variable(data_batch).cuda(device)
                label_batch = Variable(label_batch.cuda(device, async = True))
            else:
                data_batch = Variable(data_batch)
                label_batch = Variable(label_batch)

            logits = model(data_batch)
            loss = criterion(logits, label_batch)

            acc = accuracy(logits.data, label_batch)
            acc_calculator.update(acc.item(), data_batch.size(0))
            loss_calculator.update(loss.item(), data_batch.size(0))

        print('Test loss after epoch %d: %.4f, accuracy = %.2f%%'%(epoch_idx + 1, loss_calculator.average, acc_calculator.average * 100.))
        tosave['test_acc'][epoch_idx] = acc_calculator.average
        tosave['test_loss'][epoch_idx] = loss_calculator.average

        pickle.dump(tosave, open(os.path.join(output_folder, '%s.pkl'%model_name), 'wb'))
        torch.save(model.state_dict(), os.path.join(output_folder, '%s.ckpt'%model_name))

    return tosave

