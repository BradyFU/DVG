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
def define_IP(is_train=False):
    netIP = network_29layers_v2(resblock, [1, 2, 3, 4], is_train)
    netIP = torch.nn.DataParallel(netIP).cuda()
    return netIP


# define recognition network
def LightCNN_29v2(num_classes=10000, is_train=True):
    net = network_29layers_v2(resblock, [1, 2, 3, 4], is_train, num_classes=num_classes)
    net = torch.nn.DataParallel(net).cuda()
    return net

