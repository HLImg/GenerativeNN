# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/11/30 20:32:52
# @FileName:   __init__.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import torch
import torch.nn as nn

class DCGAN(nn.Module):
    def __init__(self) -> None:
        super(DCGAN, self).__init__()

        self.net_g = Generator()
        self.net_d = Discriminator()
    
    def forward(self, inp):
        """ the format for inputs on GAN Models
        Args:
            inp (dict): {
                "x" : data,
                "net_g": True or False
            }
        """
        g_train = inp['net_g']

        if g_train:
            for p in self.net_d.parameters():
                p.requires_grad = False
            return self.net_g(inp['x'])
        else:
            for p in self.net_d.parameters():
                p.requires_grad = True
            return self.net_d(inp['x'])
        

            


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1)
    