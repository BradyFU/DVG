# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2019-09-25 12:34:58
# @Breif: 
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2019-09-26 09:01:30

import torch
from .generator import Encoder, Decoder
from .light_cnn import network_29layers_v2, resblock

# define generator
def define_G(hdim=256):

    netE_nir = Encoder(hdim=hdim)
    netE_vis = Encoder(hdim=hdim)
    netG = Decoder(hdim=2*hdim)

    netE_nir = torch.nn.DataParallel(netE_nir).cuda()
    netE_vis = torch.nn.DataParallel(netE_vis).cuda()
    netG = torch.nn.DataParallel(netG).cuda()

    return netE_nir, netE_vis, netG


# define identity preserving && feature extraction net
def define_IP(cuda=True):
    netIP = network_29layers_v2(resblock, [1, 2, 3, 4], is_train=False)
    netIP = torch.nn.DataParallel(netIP).cuda()
    return netIP


# define recognition network
def LightCNN_29v2(num_classes, cuda=True):
    net = network_29layers_v2(resblock, [1, 2, 3, 4], is_train=True, num_classes=num_classes)
    net = torch.nn.DataParallel(net).cuda()
    return net


