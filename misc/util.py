# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2019-09-25 15:14:19
# @Breif: 
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2019-09-26 09:12:43

import os
import time
import argparse
import numpy as np

import torch


def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mu)

def kl_loss(mu, logvar, prior_mu=0):
    v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
    return v_kl

def reconstruction_loss(prediction, target, size_average=False):
    error = (prediction - target).view(prediction.size(0), -1)
    error = error ** 2
    error = torch.sum(error, dim=-1)

    if size_average:
        error = error.mean()
    else:
        error = error.sum()
    return error

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def save_checkpoint(model, epoch, iteration):
    model_out_path = "model/" + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
