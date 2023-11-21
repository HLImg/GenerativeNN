# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/18 12:19:52
# @FileName:  __init__.py
# @Contact :  lianghao@whu.edu.cn
# source : (https://zhuanlan.zhihu.com/p/574208925), (https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

import torch
import torch.nn as nn

class UnetVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, hiddens=[16, 32, 64, 128, 256], img_length=64) -> None:
        super(UnetVAE, self).__init__()
        
        # encoder 
        pre_dim = in_dim
        modules = []
        
        for cur_dim in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(pre_dim, cur_dim, kernel_size=3, 
                              stride=2, padding=1), 
                    nn.BatchNorm2d(cur_dim),
                    nn.ReLU()
                )
            )
            pre_dim = cur_dim
            img_length = img_length // 2
        
        self.encoder = nn.Sequential(*modules)
        
        # latent space
        self.mean_linear = nn.Linear(pre_dim * img_length * img_length, latent_dim)
        self.var_linear = nn.Linear(pre_dim * img_length * img_length, latent_dim)
        
        self.latent_dim = latent_dim
        
        # decoder
        modules = []
        self.decoder_proj = nn.Linear(latent_dim, pre_dim * img_length * img_length)
        self.decoder_inp_chw = (pre_dim, img_length, img_length)
        
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hiddens[i], hiddens[i - 1], kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hiddens[i - 1]),
                nn.ReLU()
            ))
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0], hiddens[0], kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def representise(self, mu, logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(logvar / 2)
        return mu + epsilon * std
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        
        mean = self.mean_linear(encoded)
        # TODO: 为什么是log var
        logvar = self.var_linear(encoded)
            
        # resampling
        z = self.representise(mean, logvar)
        x = self.decoder_proj(z)
        x = torch.reshape(x, (-1, *self.decoder_inp_chw))        
        decoded = self.decoder(x)
        
        return decoded, mean, logvar
    
    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_proj(z)
        x = torch.reshape(x, (-1, *self.decoder_inp_chw))
        decoded = self.decoder(x)
        return decoded