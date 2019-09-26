# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2019-09-24 10:15:14
# @Breif: dataloader for generation part
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2019-09-25 12:35:23

import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class GenDataset(data.Dataset):
    def __init__(self, img_root, list_file):
        super(GenDataset, self).__init__()

        self.img_root = img_root
        self.list_file = list_file
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ])

        self.img_list, self.make_pair_dict = self.file_reader()

    def __getitem__(self, index):
        img_line = self.img_list[index]
        img_name, label, domain_flag = img_line.strip().split(' ')

        if int(domain_flag) == 0:
            img_name_domain0 = img_name
            img_name_domain1 = self.get_pair(label, '1')
        elif int(domain_flag) == 1:
            img_name_domain0 = self.get_pair(label, '0')
            img_name_domain1 = img_name

        img_domain0 = Image.open(os.path.join(self.img_root, img_name_domain0))
        img_domain1 = Image.open(os.path.join(self.img_root, img_name_domain1))

        img_domain0 = self.transform(img_domain0)
        img_domain1 = self.transform(img_domain1)

        return {'0': img_domain0, '1': img_domain1}

    def __len__(self):
        return len(self.img_list)

    def file_reader(self):

        def dict_profile():
            return {'0': [], '1': []}

        with open(self.list_file) as file:
            img_list = file.readlines()
            img_list = [x.strip() for x in img_list]

        make_pair_dict = defaultdict(dict_profile)

        for line in img_list:
            img_name, label, domain_flag = line.strip().split(' ')
            make_pair_dict[label][domain_flag].append(img_name)

        return img_list, make_pair_dict

    def get_pair(self, label, domain_flag):
        img_name = random.choice(self.make_pair_dict[label][domain_flag])
        return img_name