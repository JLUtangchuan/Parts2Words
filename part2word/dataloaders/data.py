#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2020/11/25 20:34:57
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Data provider
'''



import json
import os
import pickle
import platform
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data



def add_vocab(i2w, w2i, word):
    idx = len(i2w)+1
    i2w[idx] = word
    w2i[word] = idx

def rotate_point(point, rotation_angle):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, 0, -sin_theta],
                                [0, 1, 0],
                                [sin_theta, 0, cos_theta]])
    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
    return rotated_point

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, shapenet_path, vocab, part_num, cfg):
        self.vocab = vocab
        # cfg for point cloud
        self.npoints = cfg.num_points
        self.pkl_path = os.path.join(shapenet_path, cfg.pkl_path)
        self.seg_num = cfg.SEG_NUM

        self.part_num = part_num

        # 读取split
        if data_split == "train":
            self.is_train = True
            file_split = cfg.data_split["train_data"]

        elif data_split == "test":
            self.is_train = False
            file_split = cfg.data_split["test_data"]

        # 读取点云数据
        self.data_dic  = {}
        with open(self.pkl_path, 'rb') as f:
            self.pkl_data = pickle.load(f)
        self.modelid_data = [k for k in self.pkl_data.keys()]

        # 取两个数据集的交集
        mids = []
        for sp in file_split:
            m = np.loadtxt(sp, dtype=str).tolist()
            for i in m:
                if i in self.modelid_data:
                    mids.append(i)
        
        # 读取vocab映射文件
        with open('/'.join((shapenet_path, 'vocab/shapenet.json')), encoding='utf-8') as f:
            self.vocab_json = json.load(f)
        self.vocab_mapping = self.vocab_json['word_to_idx']
        self.i2w = self.vocab_json['idx_to_word']
        add_vocab(self.i2w, self.vocab_mapping, '<start>')
        add_vocab(self.i2w, self.vocab_mapping, '<end>')

        # 建立字典
        self.mid_cap_data = [] #[[mid, caption], ..]
        self.mid_cap = defaultdict(list)
        caps = self.vocab_json['captions']
        for i in caps:
            self.mid_cap[i['model']].append(i['caption'])
        for i in mids:
            for j in self.mid_cap[i]:
                self.mid_cap_data.append([i, j])

        self.length = len(self.mid_cap_data)


    def __getitem__(self, index):
        model_id, caption = self.mid_cap_data[index]

        # 获取点云以及语义标注
        xyz_data, _, seg_anno_data = self.pkl_data[model_id]
        choice = np.random.choice(
            seg_anno_data.shape[0], self.npoints, replace=True)
        xyz_data_ = xyz_data[choice]
        seg_anno_data_ = seg_anno_data[choice]
        if self.is_train:
            # scale
            xyz_data_[:, :3] = xyz_data_[:, :3] * np.random.uniform(0.9, 1.1)
            # rotate
            rotate_angle = np.random.uniform(-np.pi/2, np.pi/2)
            rot_xyz = rotate_point(xyz_data_[:, :3], rotate_angle)
            xyz_data_[:, :3] = rot_xyz

        # normalize
        xyz_data_[:, 3:] = xyz_data_[:, 3:] - 0.5
        xyz_data_ = torch.from_numpy(xyz_data_).float()
        seg_anno_data_ = torch.from_numpy(seg_anno_data_).long()


        caption_ = []
        caption_.append(self.vocab_mapping['<start>'])
        caption_.extend([self.vocab_mapping[token] for token in caption])
        caption_.append(self.vocab_mapping['<end>'])
        target = torch.as_tensor(caption_, dtype=torch.long)
        return xyz_data_, target, seg_anno_data_, index, model_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    xyzrgbs, captions, semantic_labels, ids, model_ids = zip(*data)

    xyzrgbs = torch.stack(xyzrgbs, 0)
    semantic_labels = torch.stack(semantic_labels, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return {
        "shapes": xyzrgbs,
        "captions": targets,
        "semantic_labels": semantic_labels,
        "lengths": lengths,
        "ids": ids,
        "model_ids": model_ids
    }



def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt.shapenet_path, vocab, opt.K, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(vocab, batch_size, workers, opt):
    dpath = opt.data_path

    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                    batch_size, True, workers, collate_fn)
    
    return train_loader, val_loader
