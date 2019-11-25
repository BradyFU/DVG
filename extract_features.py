import os
import time
import cv2
import argparse
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from misc import *
from networks import *

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='LightCNN')
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--is_color', default=False)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--root_path', default='', type=str)
parser.add_argument('--img_list', default='', type=str)
parser.add_argument('--save_path', default='', type=str)
parser.add_argument('--batch_size', default=32, type=int)


def default_loader(path):
    img = Image.open(path).convert('L')
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, img_name))

        if self.transform is not None:
            img = self.transform(img)
        return img_name, img, target

    def __len__(self):
        return len(self.imgList)


def main():
    global args
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model = LightCNN_29v2(is_train=False)

    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading pretrained lightcnn model '{}'".format(args.weights))
            load_model(model, args.weights)

    img_list  = read_list(args.img_list)

    dataset = torch.utils.data.DataLoader(
                ImageList(root=args.root_path, fileList=args.img_list,
                    transform=transforms.Compose([
                        transforms.CenterCrop(128),
                        transforms.ToTensor(),
                    ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=1, pin_memory=True)

    model.eval()
    count = 0
    end = time.time()
    for i, (img_names, input, label) in enumerate(dataset):
        start = time.time()
        input_var = Variable(input.cuda())

        with torch.no_grad():
            features = model(input_var)

        end = time.time() - start

        for j, img_name in enumerate(img_names):
            count = count + 1
            feat = features[j, :]
            print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list), end / int(args.batch_size)))
            save_feature(args.save_path, img_name, feat.data.cpu().numpy())



def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid = open(fname, 'wb')
    fid.write(features)
    fid.close()

if __name__ == '__main__':
    main()