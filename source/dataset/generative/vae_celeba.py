# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/18 16:25:28
# @FileName:  vae_celeba.py
# @Contact :  lianghao@whu.edu.cn

import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class CelebaDataSet(Dataset):
    def __init__(self, root, img_shape=(64, 64)):
        super(CelebaDataSet, self).__init__()
        
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))
        
        self.pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        path = os.path.join(self.root, self.filenames[item])
        img = Image.open(path).convert('RGB')
        return {'lq': self.pipeline(img)}
    
    