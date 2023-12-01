# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/12/01 11:01:04
# @FileName:   gan.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import torch
import torch.nn as nn
import torch.optim as optim
from source.model.base_model import BaseModel

class GANModel(BaseModel):
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)

        self.optim_g = optim.Adam(self.net_g.net.g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.net_g.net.d.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        self.label = torch.full()
    

    def __feed__(self, data):
        self.optim_g.zero_grad()
        
