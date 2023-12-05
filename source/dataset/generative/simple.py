# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/12/05 19:15:16
# @FileName:   simple.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import os
import numpy as np

from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


class SimpleDataSet(Dataset):
    def __init__(self, root, scale_size=128, patch_size=64):
        super(SimpleDataSet, self).__init__()

        pipeline = transforms.Compose([
            transforms.transforms.Resize(scale_size),
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = datasets.ImageFolder(root, transform=pipeline)
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, item):
        return {'lq': self.dataset.__getitem__(item)}