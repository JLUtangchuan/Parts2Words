# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import importlib
import os
import logging
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from collections import defaultdict
from rich import print
from rich.progress import track
import tensorboard_logger as tb_logger

torch.backends.cudnn.benchmark = True

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir+'/..')
from utils.common_tools import multi_index_list, AverageMeter, LogCollector, draw_table
from utils.config import cfg_match as cfg
model = importlib.import_module(cfg.models)



def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    attn_embs = None
    cap_embs = None
    cap_lens = None
    k_parts = None
    model_index = None

    max_n_word = 0
    val_loss = 0.
    for i, batch_data in enumerate(data_loader):
        lengths = batch_data["lengths"]
        max_n_word = max(max_n_word, max(lengths))
    with torch.no_grad():
        for i, batch_data  in enumerate(data_loader):
            # make sure val logger is used
            # (images, captions, lengths, ids, k_part, model_ids)
            shapes = batch_data["shapes"].cuda()
            captions = batch_data["captions"].cuda()
            lengths = batch_data["lengths"]
            ids = batch_data["ids"]
            model_ids = batch_data["model_ids"]

            model.logger = val_logger
            sem_pred, point_embed, trans_feat = model.forward_pointnet(shapes)
            if model.precomp_enc_type == 'rgb':
                point_embed = torch.cat((point_embed, shapes[:,:,-3:]), dim=-1)
            part_emb, k_part = model.point2part(point_embed, sem_pred.argmax(-1))
            # compute the embeddings
            img_emb, cap_emb, cap_len = model.forward_emb(
                part_emb, captions, lengths)
            img_emb = model.mask_emb(img_emb, k_part)
            # measure accuracy and record loss
            val_loss += model.forward_loss(img_emb, cap_emb, cap_len, k_part)
            
            if img_embs is None:
                if img_emb.dim() == 3:
                    img_embs = np.zeros((len(data_loader.dataset), model.max_k_part, img_emb.size(2)))
                else:
                    img_embs = np.zeros((len(data_loader.dataset), model.max_k_part))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
                cap_lens = [0] * len(data_loader.dataset)
                k_parts = [0] * len(data_loader.dataset)
                model_index = [0] * len(data_loader.dataset)
                caption_idx_li = [0] * len(data_loader.dataset)
            # cache embeddings
            img_embs[ids,:max(k_part),:] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
            for j, nid in enumerate(ids):
                cap_lens[nid] = cap_len[j].item()
                k_parts[nid] = k_part[j]
                model_index[nid] = model_ids[j]
                caption_idx_li[nid] = captions[j]
            
            # measure elapsed time
            batch_time.update(time.time() - end, 1)
            end = time.time()

            if (1+i) % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            1+i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
    return (img_embs, cap_embs, cap_lens, k_parts, model_index, caption_idx_li), val_loss/len(data_loader)


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def shard_xattn_t2i(images, captions, caplens, k_parts, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        k = k_parts[im_start:im_end]
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.Tensor(torch.from_numpy(images[im_start:im_end]).float()).cuda()
            s = torch.Tensor(torch.from_numpy(captions[cap_start:cap_end]).float()).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_t2i(im, s, l, k, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, k_parts, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        k = k_parts[im_start:im_end]
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.Tensor(torch.from_numpy(images[im_start:im_end]).float()).cuda()
            s = torch.Tensor(torch.from_numpy(captions[cap_start:cap_end]).float()).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_i2t(im, s, l, k, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def emd_score(images, captions, caplens, k_parts, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        k = k_parts[im_start:im_end]
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> Earth Mover Distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.Tensor(torch.from_numpy(images[im_start:im_end]).float()).cuda()
            s = torch.Tensor(torch.from_numpy(captions[cap_start:cap_end]).float()).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.get_emd_score(im, s, l, k, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def ndcg(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 102))

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
        return dcg

    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
        dcg = dcg_at_n(current[i], n=k)
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg
        ndcgs.append(tmp_ndcg)
    return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))


def i2t(images, captions, caplens, model_index_image, model_index_text, sims, caption_idx_li, devocab, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap

    ranks_dic: {mid:[caption,]}
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    vocab_sz = len(devocab)

    k = 5
    topk_ = np.zeros((npts, k))
    pos_i = np.zeros((npts, k))

    ranks_dic = {}

    for index in range(npts):
        # inds = np.argsort(sims[index])[::-1]
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        mid = model_index_image[index]
        cap_idx = [i for i, modelid in enumerate(model_index_text) if mid == modelid]
        for i in cap_idx:
            rk = np.where(inds == i)[0][0]
            if rk < rank:
                # 找到同一个modelid的caption的rank最低者
                rank = rk
            if rk < k:
                topk_[index, rk] = 1.0
        max_num = min(len(cap_idx), k)
        pos_i[index, :max_num] = 1.0

        ranks[index] = rank
        top1[index] = inds[0]
        # top5 caption
        if return_ranks:
            caps = []
            for cap in multi_index_list(caption_idx_li, inds[:k]):
                caps.append(" ".join([devocab[str(j.item())] for j in cap
                        if j <= vocab_sz and j > 0]))
            ranks_dic[mid] = caps

    # Compute metrics
    ndcg_ = 100.0 * ndcg(pos_i, topk_, k)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ndcg_, ranks_dic)
    else:
        return (r1, r5, r10, medr, meanr, ndcg_)


def t2i(images, captions, caplens, model_index_image, model_index_text, sims, caption_idx_li, devocab, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap

    return_ranks "mid":[(caption, [top5 modelid]), ()]
    """
    # npts = images.shape[0]
    npts = captions.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    vocab_sz = len(devocab)

    k = 5
    topk_ = np.zeros((npts, k))
    pos_i = np.zeros((npts, k))
    # =>(shape_len, caption_len)
    sims = sims.T
    # =>(caption_len, shape_len)
    ranks_dic = defaultdict(list)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        
        ranks[index] = np.where(np.array(multi_index_list(model_index_image, inds)) == model_index_text[index])[0][0]
        top1[index] = inds[0]
        rk = ranks[index]
        if rk < k:
            topk_[index, int(rk)] = 1.0
        pos_i[index, 0] = 1.0
        if return_ranks:
            # caption
            cap = " ".join([devocab[str(j.item())] for j in caption_idx_li[index]
                if j <= vocab_sz and j > 0])
            # top5 mid
            mids = multi_index_list(model_index_image, inds[:k])
            ranks_dic[model_index_text[index]].append((cap, mids))


    # Compute metrics
    ndcg_ = 100.0 * ndcg(pos_i, topk_, k)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ndcg_, ranks_dic)
    else:
        return (r1, r5, r10, medr, meanr, ndcg_)

def del_repeat(img_embs, cap_embs, cap_lens, k_parts, model_index):
    _ , uidx, uinv = np.unique(model_index, return_index=True, return_inverse=True)
    img_embs = img_embs[uidx]
    k_parts = multi_index_list(k_parts, uidx)
    model_index_image = multi_index_list(model_index, uidx)
    model_index_text = model_index
    return img_embs, cap_embs, cap_lens, k_parts, model_index_image, model_index_text, uinv

def sem_seg(data_loader, model, opt):
    accuracy = AverageMeter()
    sem_loss = AverageMeter()
    model.val_start()
    val_logger = LogCollector()

    with torch.no_grad():
        for i, val_data in enumerate(data_loader):
            model.logger = val_logger
            shapes = val_data["shapes"]
            semantic_labels = val_data["semantic_labels"]
            if torch.cuda.is_available():
                shapes, semantic_labels = shapes.to(dtype=torch.float).cuda(), semantic_labels.to(dtype=torch.long).cuda()
            sem_pred, _, trans_feat = model.forward_pointnet(shapes)
            semantic_loss = model.forward_semantic_loss(sem_pred, semantic_labels, trans_feat)

            pred_choice = sem_pred.view(-1, opt.SEG_NUM).data.max(1)[1]
            correct = pred_choice.eq(semantic_labels.view(-1, 1)[:, 0].data).cpu().to(dtype=torch.float).mean()
            accuracy.update(correct.item(), shapes.size(0)*opt.num_points)
            sem_loss.update(semantic_loss.item(), shapes.size(0))
    return accuracy, sem_loss

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(opt, val_loader, model, devocab, epoch, return_ranks=False):
    # Semantic segmantation validation
    acc, loss = sem_seg(val_loader, model, opt)
    draw_table({
            "Epoch": epoch,
            "Acc": acc.avg,
            "Sem loss": loss.avg
        })
    currscore = acc.avg
    # Matching task validation
    if epoch >= opt.stage_1_epoch:
        # compute the encoding for all the validation images and captions
        (img_embs, cap_embs, cap_lens, k_parts, model_index, caption_idx_li), val_loss = encode_data(
            model, val_loader, 56, logging.info)

        # del repeat shape embedding
        img_embs, cap_embs, cap_lens, k_parts, model_index_image, model_index_text, uinv = del_repeat(
            img_embs, cap_embs, cap_lens, k_parts, model_index)
        
        start = time.time()
        if opt.matching_method == 'scan':
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, k_parts, opt, shard_size=opt.batch_size)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, k_parts, opt, shard_size=opt.batch_size)
            else:
                raise NotImplementedError
        elif opt.matching_method == 'emd':
                sims = emd_score(img_embs, cap_embs, cap_lens, k_parts, opt, shard_size=opt.batch_size)
            
        
        end = time.time()
        print("calculate similarity time:", end-start)


        # caption retrieval
        i2t_res = i2t(
            img_embs, cap_embs,cap_lens, model_index_image, model_index_text, sims, caption_idx_li, devocab, return_ranks=return_ranks)
        if return_ranks:
            (r1, r5, r10, medr, meanr, ndcg, i2t_dic) = i2t_res
        else:
            (r1, r5, r10, medr, meanr, ndcg) = i2t_res

        logging.info("Image to text: %.2f, %.2f, %.2f, %.1f, %.1f, %.2f" %
                    (r1, r5, r10, medr, meanr, ndcg))
        # image retrieval
        t2i_res = t2i(
            img_embs, cap_embs, cap_lens, model_index_image, model_index_text, sims, caption_idx_li, devocab, return_ranks=return_ranks)

        if return_ranks:
            (r1i, r5i, r10i, medri, meanr, ndcgi, t2i_dic) = t2i_res
        else:
            (r1i, r5i, r10i, medri, meanr, ndcgi) = t2i_res
        
        logging.info("Text to image: %.2f, %.2f, %.2f, %.1f, %.1f, %.2f" %
                    (r1i, r5i, r10i, medri, meanr, ndcgi))
        
        draw_table({
            "r1": r1,
            "r5": r5,
            "ndcg": ndcg,
            "r1i": r1i,
            "r5i": r5i,
            "ndcgi": ndcgi,
            "Val loss": val_loss
        })
        # sum of recalls to be used for early stopping
        currscore = ndcg + ndcgi

        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('ndcg', ndcg, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('ndcgi', ndcgi, step=model.Eiters)
        tb_logger.log_value('val_loss', val_loss, step=model.Eiters)
        # tb_logger.log_value('rsum', currscore, step=model.Eiters)

        # save
        if return_ranks:
            saved_dic = {
                "i2t_dic": i2t_dic,
                "t2i_dic": t2i_dic
            }
            with open("result.json", "w") as f:
                json.dump(saved_dic, f)
                print("Save json file successfully")
    return currscore