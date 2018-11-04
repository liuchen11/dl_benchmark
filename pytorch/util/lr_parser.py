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

>>> cosine scheduling
name=cosine_drop,max_value=$VALUE$,min_value=$VALUE$,unit_period=$VALUE$,period_scale=$1.$,value_decay=$1.$
'''

def parse_lr(policy, epoch_num):

    if policy['name'].lower() in ['c', 'constant']:
        lr_func = constant(value = float(policy['value']))
    elif policy['name'].lower() in ['exp_decay']:
        lr_func = exp_decay(start_value = policy['start_value'], decay_ratio = policy['decay_ratio'], decay_freq = int(policy['decay_freq']))
    elif policy['name'].lower() in ['exp_drop']:
        milestones = list(map(int, policy['milestones'].split('_'))) if isinstance(policy['milestones'], str) else [policy['milestones'],]
        lr_func = exp_drop(start_value = policy['start_value'], decay_ratio = policy['decay_ratio'], milestones = milestones)
    elif policy['name'].lower() in ['cosine_drop']:
        policy['period_scale'] = 1. if not 'period_scale' in policy else policy['period_scale']
        policy['value_decay'] = 1. if not 'value_decay' in policy else policy['value_decay']
        lr_func = cosine_drop(max_value = float(policy['max_value']), min_value = float(policy['min_value']),
            unit_period = int(policy['unit_period']), period_scale = float(policy['period_scale']), value_decay = float(policy['value_decay']))
    elif policy['name'].lower() in ['h', 'help']:
        print(instructions)
        exit(0)
    else:
        raise ValueError('Unrecognized learning rate policy: %s'%policy)

    print('LR policy: %s --'%policy['name'])
    for key in policy:
        print('%s: %s'%(key, policy[key]))
    print('----------------')

    return lr_func

def constant(value):

    return lambda x: value

def exp_decay(start_value, decay_ratio, decay_freq):

    return lambda x: start_value * decay_ratio ** (float(x) / decay_freq)

class exp_drop_mapper:

    def __init__(self, start_value, decay_ratio, milestones):
        self.start_value = start_value
        self.decay_ratio = decay_ratio
        self.milestones = milestones

    def __call__(self, x):
        lr = self.start_value
        for ms in self.milestones:
            if x >= ms:
                lr *= self.decay_ratio
        return lr

def exp_drop(start_value, decay_ratio, milestones):

    return exp_drop_mapper(start_value, decay_ratio, milestones)

class cosine_drop_mapper:

    def __init__(self, max_value, min_value, unit_period, period_scale = 1., value_decay = 1.):
        self.max_value = max_value
        self.min_value = min_value
        self.unit_period = unit_period
        self.period_scale = period_scale
        self.value_decay = value_decay

    def __call__(self, x):
        current_unit = self.unit_period
        max_v = self.max_value
        min_v = self.min_value
        while x >= current_unit:
            x -= current_unit
            current_unit *= self.period_scale
            max_v *= self.value_decay
            min_v *= self.value_decay

        result = min_v + float(1. + np.cos(float(x) / current_unit * np.pi)) / 2. * (max_v - min_v)
        return result

def cosine_drop(max_value, min_value, unit_period, period_scale = 1., value_decay = 1.):

    return cosine_drop_mapper(max_value, min_value, unit_period, period_scale, value_decay)
