#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/10/12 21:24:36
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   yaml
'''

import argparse
import yaml
import os

def get_parser(task = None):
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation & Matching Parameter')
    parser.add_argument('--config', type=str, default='config/baseline1012.yaml', help='path to config file')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if task is None:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
    else:
        for key in config:
            if key == task or key == 'GENERAL':
                for k, v in config[key].items():
                    setattr(args_cfg, k, v)
        

    return args_cfg


# cfg = get_parser()
cfg_seg = get_parser(task = "SEGMENTATION")
cfg_match = get_parser(task = "MATCHING")

setattr(cfg_seg, 'save_dir', os.path.join(cfg_seg.home, cfg_seg.save_dir))
setattr(cfg_match, 'save_dir', os.path.join(cfg_match.home, cfg_match.save_dir))

setattr(cfg_seg, 'shapenet_path', os.path.join(cfg_seg.home, cfg_seg.shapenet_path))
setattr(cfg_match, 'shapenet_path', os.path.join(cfg_match.home, cfg_match.shapenet_path))

# setattr(cfg_seg, 'h5_path', os.path.join(cfg_seg.shapenet_path, cfg_seg.h5_path))
setattr(cfg_match, 'resume', os.path.join(cfg_match.save_dir, cfg_match.resume))

if __name__ == '__main__':
    print(cfg_seg, cfg_match)
