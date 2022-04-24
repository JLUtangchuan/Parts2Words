#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   common_tools.py
@Time    :   2021/10/13 14:00:01
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   None
'''

import datetime
import functools
import logging
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

def timer(func):
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('%s took %f second' % (func.__name__, end_time - start_time))
        return res
    return time_wrapper

def check_tensor(mat, name = ''):
    if torch.isnan(mat).sum().item() != 0:
        raise ValueError(f'Error {name} : {torch.isnan(mat).sum().item()}')

def multi_index_list(li:list, idx):
    return [li[i] for i in idx]


def draw_table(dic):
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim", width=12)
    
    for k in dic.keys():
        table.add_column(k)
    val = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    val += ["%.2f" % v for v in dic.values()]
    table.add_row(*val)
    console.print(table)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None
    
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, os.path.join(prefix, filename))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, '/'.join((prefix, filename)))
            if is_best:
                shutil.copyfile('/'.join((prefix, filename)), '/'.join((prefix, 'model_best.pth.tar')))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)
            

