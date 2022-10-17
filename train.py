# -*- coding=utf-8 -*-
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/04/24 14:00:04
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Train
'''

import importlib
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorboard_logger as tb_logger
import torch
from rich import print
from rich.progress import track

from utils.common_tools import draw_table, set_seed, save_checkpoint, AverageMeter, LogCollector
from utils.config import cfg_match as cfg
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
    log_pth = '/'.join((opt.save_dir, 'validate.log'))
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
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            print("Test Loader Result")
            # validate(opt, val_loader, model, devocab, epoch=start_epoch, return_ranks=False)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    # lr update scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model.optimizer, gamma=0.99)
    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"Epoch: {epoch}")
        
        # train for one epoch
        train(opt, train_loader, model, epoch)
        scheduler.step()

        # evaluate on validation set
        if (epoch+1) % 1 == 0:
            if opt.eval_when_training:
                rsum = validate(opt, val_loader, model, devocab, epoch=epoch)
            else:
                rsum = epoch
            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            if not os.path.exists(opt.model_name):
                os.mkdir(opt.model_name)
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
        

def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    
    for i, train_data in track(enumerate(train_loader), total=len(train_loader)):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end, 1)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(train_data, epoch=epoch)

        # measure elapsed time
        batch_time.update(time.time() - end, 1)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

if __name__ == '__main__':
    main()
