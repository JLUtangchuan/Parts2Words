#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   val.py
@Time    :   2022/04/24 14:00:44
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   None
'''


import importlib
import json
import logging
import os
import shutil
import sys
import time
# from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorboard_logger as tb_logger
import torch
from rich import print
from rich.progress import track


from utils.common_tools import draw_table, set_seed, save_checkpoint, AverageMeter, LogCollector
from utils.config import cfg_match as cfg
from utils.vocab import Vocabulary, deserialize_vocab
dataloader_module = importlib.import_module(cfg.dataloader)
model_module = importlib.import_module(cfg.models)
eval_module = importlib.import_module(cfg.evals)
validate = eval_module.validate


set_seed()

def main():
    opt = cfg
    print(opt)

    # Load Vocabulary Wrapper
    with open('/'.join((opt.shapenet_path, 'vocab', 'shapenet.json'))) as f:
        js_data = json.load(f)
        vocab = js_data['word_to_idx']
        devocab = js_data['idx_to_word']
    opt.vocab_size = len(vocab) + 3  # <start> <end> 0

    # opt
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    opt.logger_name = '/'.join((opt.save_dir, 'runs/runX/log'))
    opt.model_name = '/'.join((opt.save_dir, 'runs/runX/checkpoint'))
    opt.data_path = opt.save_dir

    # logging
    log_pth = '/'.join((opt.save_dir, 'validate_val.log'))
    logging.basicConfig(filename=log_pth, filemode='a+', 
                        format="%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s|%(message)s", 
                        level=logging.INFO)
    
    tb_logger.configure(opt.logger_name, flush_secs=5)
    # Load data loaders
    train_loader, val_loader = dataloader_module.get_loaders(vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = model_module.SCAN(opt)

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    for i in range(68,70):
        resume = opt.save_dir + '/runs/runX/checkpoint/checkpoint_'+ str(i) +'.pth.tar'
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                    .format(resume, start_epoch, best_rsum))
            print("Test Loader Result")
            validate(opt, val_loader, model, devocab, epoch=start_epoch, return_ranks=False)
    else:
        print("=> no checkpoint found at '{}'".format(resume))
if __name__ == '__main__':
    main()
