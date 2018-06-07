# parse learning rate scheduling

import numpy as np

def lr_parser(policy, epoch_num):

    policy_parts = policy.split(',')
    if policy_parts[0].lower() in ['c']:
        lr_list = constant(value = int(policy_parts[1]), epoch_num = epoch_num)
    elif policy_parts[0].lower() in ['exp_decay']:
        lr_list = exp_decay(start_value = float(policy_parts[1]), decay_ratio = float(policy_parts[2]),
            decay_freq = int(policy_parts[3]), epoch_num = epoch_num)
    elif policy_parts[0].lower() in ['exp_drop']:
        lr_list = exp_drop(start_value = float(policy_parts[1]), decay_ratio = float(policy_parts[2]),
            milestones = list(map(int, policy_parts[3:])), epoch_num = epoch_num)
    else:
        raise ValueError('Unrecognized learning rate policy: %s'%policy)

    assert len(lr_list) == epoch_num, 'Error: len(lr_list) = %d, epoch_num = %d'%(len(lr_list), epoch_num)

    return lr_list

def constant(value, epoch_num):
    return np.array([value,] * epoch_num)

def exp_decay(start_value, decay_ratio, decay_freq, epoch_num):

    end_value = start_value * decay_ratio ** (float(epoch_num - 1) / decay_freq)
    lr_list = np.logspace(np.log10(start_value), np.log10(end_value), epoch_num)

    return lr_list

def exp_drop(start_value, decay_ratio, milestones, epoch_num):

    milestones = list(sorted(milestones))
    milestones = [0, ] + milestones + [epoch_num,]

    lr_list = []
    value = start_value
    for seg_start, seg_end in zip(milestones[:-1], milestones[1:]):
        lr_list = lr_list + [value,] * (seg_end - seg_start)
        value = value * decay_ratio

    return lr_list
