import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from data import *
from networks import *
from misc import *

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--all_epochs', default=500, type=int)
parser.add_argument('--pre_epoch',  default=0, type=int, help='train from previous model')
parser.add_argument('--hdim', default=128, type=int, help='dim of the latent code')
parser.add_argument('--out_path', default='results/', type=str, help='folder to sive output images')
parser.add_argument('--print_freq', type=int, default=20, help='print frequency')

parser.add_argument('--test_epoch', default=100, type=int)
parser.add_argument('--save_epoch', default=1, type=int)

parser.add_argument('--ip_model', default='', type=str, help='path of identity preserving model')
parser.add_argument('--img_root', default='', type=str,)
parser.add_argument('--train_list', default='', type=str)

parser.add_argument('--lambda_mmd',  default=10, type=float)
parser.add_argument('--lambda_ip',  default=10, type=float)
parser.add_argument('--lambda_pair',  default=0.5, type=float)

def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # generator
    netE_nir, netE_vis, netG = define_G(hdim=args.hdim)

    if args.pre_epoch:
        print('load pretrained model %d' % args.pre_epoch)
        load_model(netE_nir, './model/netE_nir_model_epoch_%d_iter_0.pth' % args.pre_epoch)
        load_model(netE_vis, './model/netE_vis_model_epoch_%d_iter_0.pth' % args.pre_epoch)
        load_model(netG, './model/netG_model_epoch_%d_iter_0.pth' % args.pre_epoch)

    # IP net
    netIP = define_IP()

    print("=> loading pretrained identity preserving model '{}'".format(args.ip_model))
    checkpoint = torch.load(args.ip_model)

    pretrained_dict = checkpoint['state_dict']
    model_dict = netIP.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netIP.load_state_dict(model_dict)

    for param in netIP.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()), lr=args.lr)

    # criterion
    criterionL2 = torch.nn.MSELoss().cuda()

    train_loader = torch.utils.data.DataLoader(
            GenDataset(img_root=args.img_root, list_file=args.train_list),
            batch_size=args.batch_size, shuffle=True, 
            num_workers=args.workers, pin_memory=True)

    # train
    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.all_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        netE_nir.train()
        netE_vis.train()
        netG.train()
        netIP.eval()

        start_time = time.time()
        for iteration, batch in enumerate(train_loader):
            data_time.update(time.time() - start_time)

            img_nir = Variable(batch['0'].cuda())
            img_vis = Variable(batch['1'].cuda())

            img = torch.cat((img_nir, img_vis), 1)

            # encoder forward
            mu_nir, logvar_nir = netE_nir(img_nir)
            mu_vis, logvar_vis = netE_vis(img_vis)

            # reparameterization
            z_nir = reparameterize(mu_nir, logvar_nir)
            z_vis = reparameterize(mu_vis, logvar_vis)

            # generator
            rec = netG(torch.cat((z_nir, z_vis), dim=1))

            # vae loss
            loss_rec = reconstruction_loss(rec, img, True) / 2.0
            loss_kl = (kl_loss(mu_nir, logvar_nir).mean() + kl_loss(mu_vis, logvar_vis).mean()) / 2.0

            # mmd loss
            loss_mmd = args.lambda_mmd * torch.abs(z_nir.mean(dim=0) - z_vis.mean(dim=0)).mean()

            # identity preserving loss
            rec_nir = rec[:, 0:3, :, :]
            rec_vis = rec[:, 3:6, :, :]

            nir_fc = netIP(rgb2gray(img_nir))
            vis_fc = netIP(rgb2gray(img_vis))
            rec_nir_fc = netIP(rgb2gray(rec_nir))
            rec_vis_fc = netIP(rgb2gray(rec_vis))

            nir_fc = F.normalize(nir_fc, p=2, dim=1)
            vis_fc = F.normalize(vis_fc, p=2, dim=1)
            rec_nir_fc = F.normalize(rec_nir_fc, p=2, dim=1)
            rec_vis_fc = F.normalize(rec_vis_fc, p=2, dim=1)

            loss_ip = args.lambda_ip * (criterionL2(rec_nir_fc, nir_fc.detach()) + criterionL2(rec_vis_fc, vis_fc.detach())) / 2.0
            loss_pair = args.lambda_pair * criterionL2(rec_nir_fc, rec_vis_fc)

            # 
            if epoch < 2:
                loss = loss_rec + 0.01 * loss_kl + 0.01 * loss_mmd + 0.01 * loss_ip + 0.01 * loss_pair
            else:
                loss = loss_rec + loss_kl + loss_mmd + loss_ip + loss_pair

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if iteration % args.print_freq == 0:
                info = '====> Epoch: [{:0>4d}][{:3d}/{:3d}] Batch_time: {:4.3f} Data_time: {:4.3f} | '.format(
                    epoch, iteration, len(train_loader), batch_time.avg, data_time.avg)

                info += 'Loss: rec: {:4.3f} kl: {:4.3f} mmd: {:4.3f} ip: {:4.3f} pair: {:4.3f}'.format(
                    loss_rec.item(), loss_kl.item(), loss_mmd.item(), loss_ip.item(), loss_pair.item())

                print(info)

        # test
        if (epoch % args.test_epoch is 0 or epoch == 1):
            noise = torch.zeros(args.batch_size, args.hdim).normal_(0, 1)
            noise = torch.cat((noise, noise), dim=1)
            noise = noise.cuda()

            fake = netG(noise)

            vutils.save_image(img_nir.data, '{}/Epoch_{:03d}_img_nir.png'.format(args.out_path, epoch))
            vutils.save_image(img_vis.data, '{}/Epoch_{:03d}_img_vis.png'.format(args.out_path, epoch))
            vutils.save_image(rec_nir.data, '{}/Epoch_{:03d}_rec_nir.png'.format(args.out_path, epoch))
            vutils.save_image(rec_vis.data, '{}/Epoch_{:03d}_rec_vis.png'.format(args.out_path, epoch))
            vutils.save_image(fake[:, 0:3, :, :].data, '{}/Epoch_{:03d}_fake_nir.png'.format(args.out_path, epoch))
            vutils.save_image(fake[:, 3:6, :, :].data, '{}/Epoch_{:03d}_fake_vis.png'.format(args.out_path, epoch))

        # save models
        if (epoch % args.save_epoch == 0 or epoch == 1):
            save_checkpoint(netE_nir, epoch, 0, 'netE_nir_')
            save_checkpoint(netE_vis, epoch, 0, 'netE_vis_')
            save_checkpoint(netG, epoch, 0, 'netG_')


if __name__ == "__main__":
    main()