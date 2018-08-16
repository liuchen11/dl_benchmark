# parse learning rate scheduling

import numpy as np

instructions = '''
instructions for setting a learning rate schedule
>>> constant learning rate: lr = v
name=constant,value=$VALUE$

>>> exponentially decaying learning rate, lr = v * ratio ** (n / decay_freq)
name=exp_decay,start_value=$VALUE$,decay_ratio=$VALUE$,decay_freq=$VALUE$

>>> exponentially dropping learning rate, each time meets a milestone, decrease the lr
name=exp_drop,start_value=$VALUE$,decay_ratio=$VALUE$,milestones=$VALUE1_VALUE2_...._VALUE$
'''

def parse_lr(policy, epoch_num):

    if policy['name'].lower() in ['c', 'constant']:
        lr_list = constant(value = policy['value'], epoch_num = epoch_num)
    elif policy['name'].lower() in ['exp_decay']:
        lr_list = exp_decay(start_value = policy['start_value'], decay_ratio = policy['decay_ratio'],
            decay_freq = int(policy['decay_freq']), epoch_num = epoch_num)
    elif policy['name'].lower() in ['exp_drop']:
        milestones = list(map(int, policy['milestones'].split('_'))) if isinstance(policy['milestones'], str) else [policy['milestones'],]
        lr_list = exp_drop(start_value = policy['start_value'], decay_ratio = policy['decay_ratio'],
            milestones = milestones, epoch_num = epoch_num)
    elif policy['name'].lower() in ['h', 'help']:
        print(instructions)
        exit(0)
    else:
        raise ValueError('Unrecognized learning rate policy: %s'%policy)

    assert len(lr_list) == epoch_num, 'Error: len(lr_list) = %d, epoch_num = %d'%(len(lr_list), epoch_num)

    print('LR policy: %s --'%policy['name'])
    for key in policy:
        print('%s: %s'%(key, policy[key]))
    print('----------------')

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
