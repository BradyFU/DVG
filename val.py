import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable

from data import *
from networks import *
from misc import *

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0,1', type=str)
parser.add_argument('--hdim', default=128, type=int)
parser.add_argument('--pre_model', default='./model/netG_model_epoch_50_iter_0.pth', type=str)
parser.add_argument('--out_path_1', default='fake_images/nir_noise/', type=str)
parser.add_argument('--out_path_2', default='fake_images/vis_noise/', type=str)


def main():
    global opt, model
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.out_path_1):
        os.makedirs(args.out_path_1)

    if not os.path.exists(args.out_path_2):
        os.makedirs(args.out_path_2)

    # generator
    netE_nir, netE_vis, netG = define_G(hdim=args.hdim)
    load_model(netG, args.pre_model)
    netG.eval()

    num = 0
    for n in range(1000):
        noise = torch.zeros(100, args.hdim).normal_(0, 1)
        noise = torch.cat((noise, noise), dim=1)
        noise = Variable(noise).cuda()

        fake = netG(noise)

        nir = fake[:, 0:3, :, :].data.cpu().numpy()
        vis = fake[:, 3:6, :, :].data.cpu().numpy()

        for i in range(nir.shape[0]):
            num = num + 1
            save_img = nir[i, :, :, :]
            save_img = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(save_img)
            save_name = str(num) + '.jpg'
            output.save(os.path.join(args.out_path_1, save_name))

            save_img = vis[i, :, :, :]
            save_img = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(save_img)
            save_name = str(num) + '.jpg'
            output.save(os.path.join(args.out_path_2, save_name))

        print(num)


if __name__ == "__main__":
    main()