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
import torchvision.transforms as transforms

from misc import *
from data import *
from networks import *


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='LightCNN')
parser.add_argument('--num_classes', default=725, type=int)

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=15, type=int)

parser.add_argument('--pre_epoch',  default=0, type=int, help='train from previous model')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

parser.add_argument('--weights', default='', type=str)
parser.add_argument('--img_root', default='', type=str)
parser.add_argument('--train_list', default='', type=str)
parser.add_argument('--fake_path', default='', type=str, help='path to save fake images')
parser.add_argument('--fake_num', default=100000, type=int)


def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    # lightcnn
    model = LightCNN_29v2(num_classes=args.num_classes)

    # load pre trained model
    if args.pre_epoch:
        print('load pretrained model %d' % args.pre_epoch)
        load_model(model, './model/lightCNN_model_epoch_%d_iter_0.pth' % args.pre_epoch)
    else:
        # load pretrained lightcnn
        print("=> loading pretrained lightcnn model '{}'".format(args.weights))
        checkpoint = torch.load(args.weights)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # dataset
    image_dataset = SeparateImageList(args.img_root, args.train_list, args.fake_path, args.fake_num)

    train_real_idx, train_fake_idx = image_dataset.get_idx()
    batch_sampler = SeparateBatchSampler(train_real_idx, train_fake_idx, batch_size=args.batch_size, ratio=0.5)
    
    # real and fake training data
    train_loader = torch.utils.data.DataLoader(
        image_dataset,
        num_workers=args.workers,
        batch_sampler=batch_sampler)

    # real training data
    val_loader = torch.utils.data.DataLoader(
        ImageList(root=args.img_root, fileList=args.train_list),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    '''
    Stage I: model pretrained for last fc2 parameters
    '''
    params_pretrain = []
    for name, value in model.named_parameters():
        if 'fc2_' in name:
            params_pretrain += [{'params': value, 'lr': 10 * args.lr}]

    # optimizer
    optimizer_pretrain = torch.optim.SGD(params_pretrain, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, 6):
        pre_train(val_loader, model, criterion, optimizer_pretrain, epoch)

        save_checkpoint(model, epoch, 0, 'lightCNN_pretrain_')


    '''
    Stage II: model finetune for full network
    '''
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    prec1 = validate(val_loader, model, criterion)

    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, args.step_size, optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        prec1 = validate(val_loader, model, criterion)

        save_checkpoint(model, epoch, 0, 'lightCNN_')




# pretrain for the last fc2 parameters
def pre_train(val_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for i, data in enumerate(val_loader):
        input_var = Variable(data['img'].cuda())
        target_var = Variable(data['label'].cuda())

        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1.item(), input_var.size(0))
        top5.update(prec5.item(), input_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:

            info = '====> Epoch: [{:0>3d}][{:3d}/{:3d}] | '.format(epoch, i, len(val_loader))
            info += 'Loss: real_ce: {:4.3f} ({:4.3f}) | '.format(losses.val, losses.avg)
            info += 'Prec@1: {:4.3f} ({:4.3f}) Prec@5: {:4.3f} ({:4.3f})'.format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_real_ce = AverageMeter()
    losses_real_mmd = AverageMeter()
    losses_fake_mmd = AverageMeter()

    model.train()

    for i, data in enumerate(train_loader):
        input_var = Variable(data['img'].cuda())
        target_var = Variable(data['label'].cuda())
        domain_var = Variable(data['domain_flag'].cuda())

        # compute output
        output, fc = model(input_var)

        # select real nir and vis data
        idx_real = torch.nonzero(target_var.data != -1)
        idx_real = torch.autograd.Variable(idx_real[:, 0])

        output_real = torch.index_select(output, 0, idx_real)
        fc_real = torch.index_select(fc, 0, idx_real)
        label_real = torch.index_select(target_var, 0, idx_real)
        domain_real = torch.index_select(domain_var, 0, idx_real)

        loss_real_ce = criterion(output_real, label_real)

        # select domain of real data
        idx_nir_real = torch.nonzero(domain_real.data != 1)
        idx_nir_real = torch.autograd.Variable(idx_nir_real[:, 0])
        fc_nir_real = torch.index_select(fc_real, 0, idx_nir_real)

        idx_vis_real = torch.nonzero(domain_real.data != 0)
        idx_vis_real = torch.autograd.Variable(idx_vis_real[:, 0])
        fc_vis_real = torch.index_select(fc_real, 0, idx_vis_real)

        loss_real_mmd = MMD_Loss(fc_nir_real, fc_vis_real)

        # select fake data
        idx_fake = torch.nonzero(target_var.data == -1)
        idx_fake = torch.autograd.Variable(idx_fake[:, 0])

        fc_fake = torch.index_select(fc, 0, idx_fake)
        domain_fake = torch.index_select(domain_var, 0, idx_fake)

        # select domain of fake data
        idx_nir_fake = torch.nonzero(domain_fake.data != 1)
        idx_nir_fake = torch.autograd.Variable(idx_nir_fake[:, 0])
        fc_nir_fake = torch.index_select(fc_fake, 0, idx_nir_fake)

        idx_vis_fake = torch.nonzero(domain_fake.data != 0)
        idx_vis_fake = torch.autograd.Variable(idx_vis_fake[:, 0])
        fc_vis_fake = torch.index_select(fc_fake, 0, idx_vis_fake)

        loss_fake_mmd = MMD_Loss(fc_nir_fake, fc_vis_fake)

        loss_HFR = loss_real_ce + 0.001 * loss_real_mmd + 0.001 * loss_fake_mmd

        optimizer.zero_grad()
        loss_HFR.backward(retain_graph=True)
        optimizer.step()

        # measure accuracy and record loss
        losses_real_ce.update(loss_real_ce.item(), output_real.size(0))
        losses_real_mmd.update(loss_real_mmd.item(), 1)
        losses_fake_mmd.update(loss_fake_mmd.item(), 1)

        prec1, prec5 = accuracy(output_real.data, label_real.data, topk=(1, 5))
        top1.update(prec1.item(), output_real.size(0))
        top5.update(prec5.item(), output_real.size(0))

        if i % args.print_freq == 0:
            info = '====> Epoch: [{:0>3d}][{:3d}/{:3d}] | '.format(epoch, i, len(train_loader))

            info += 'Loss: real_ce: {:4.3f} ({:4.3f}) real_mmd: {:4.3f} ({:4.3f}) fake_mmd: {:4.3f} ({:4.3f}) | '.format(
                losses_real_ce.val, losses_real_ce.avg, losses_real_mmd.val, losses_real_mmd.avg, losses_fake_mmd.val, losses_fake_mmd.avg)

            info += 'Prec@1: {:4.3f} ({:4.3f}) Prec@5: {:4.3f} ({:4.3f})'.format(top1.val, top1.avg, top5.val, top5.avg)

            print(info)


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for i, data in enumerate(val_loader):
        input_var = Variable(data['img'].cuda())
        target_var = Variable(data['label'].cuda())

        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1.item(), input_var.size(0))
        top5.update(prec5.item(), input_var.size(0))

    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))

    return top1.avg


if __name__ == '__main__':
    main()
