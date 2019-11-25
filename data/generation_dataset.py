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

        self.img_domain0_list, self.make_pair_dict = self.file_reader()

    def __getitem__(self, index):
        img_line = self.img_domain0_list[index]
        img_name_domain0, label, _ = img_line.strip().split(' ')

        img_name_domain1 = self.get_pair(label, '1')

        img_domain0 = Image.open(os.path.join(self.img_root, img_name_domain0))
        img_domain1 = Image.open(os.path.join(self.img_root, img_name_domain1))

        img_domain0 = self.transform(img_domain0)
        img_domain1 = self.transform(img_domain1)

        return {'0': img_domain0, '1': img_domain1}

    def __len__(self):
        return len(self.img_domain0_list)

    def file_reader(self):

        def dict_profile():
            return {'0': [], '1': []}

        with open(self.list_file) as file:
            img_list = file.readlines()
            img_list = [x.strip() for x in img_list]

        make_pair_dict = defaultdict(dict_profile)
        img_domain0_list = []

        for line in img_list:
            img_name, label, domain_flag = line.strip().split(' ')
            make_pair_dict[label][domain_flag].append(img_name)

            if domain_flag == '0':
                img_domain0_list.append(line)

        return img_domain0_list, make_pair_dict

    def get_pair(self, label, domain_flag):
        img_name = random.choice(self.make_pair_dict[label][domain_flag])
        return img_name
