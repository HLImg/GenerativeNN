# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/12/05 16:39:38
# @File    :   basic_unet_ddpm.py
# @Contact   :   lianghao@whu.edu.cn

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, channels, size):
        super(Transformer, self).__init__()
        
        self.size = size
        self.channels = channels
        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        
        self.ffn = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        
        attention_value = self.ffn(attention_value) + attention_value
        
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super(DoubleConv, self).__init__()
        
        self.residual = residual
        
        if not mid_ch:
            mid_ch = out_ch
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_ch)
        )
    
    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)
        

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim=256):
        super(DownSample, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, residual=False)
        )
        
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_ch)
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        embed_time = self.embed_layer(t)[:, :, None, None].repeat(-1, 1, x.shape[-2], x.shape[-1])
        return x + embed_time
    

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim=256):
        super(UpSample, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2, residual=False)
        )
        
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_ch)
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        embed_time = self.embed_layer(t)[:, :, None, None].repeat(-1, 1, x.shape[-2], x.shape[-1])
        return x + embed_time 

class DDPMUnet(nn.Module):
    def __init__(self, in_ch, out_ch, is_cpu=True, embed_time_dim=256):
        super(DDPMUnet, self).__init__()
        
        self.cpu = is_cpu
        self.embed_time_dim = embed_time_dim
        
        self.inc = DoubleConv(in_ch, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 256)
        
        self.sa_1 = Transformer(128, 32)
        self.sa_2 = Transformer(256, 16)
        self.sa_3 = Transformer(256, 8)
        self.sa_4 = Transformer(128, 16)
        self.sa_5 = Transformer(64, 32)
        self.sa_6 = Transformer(64, 64)
        
        self.bot_1 = DoubleConv(256, 512)
        self.bot_2 = DoubleConv(512, 512)
        self.bot_3 = DoubleConv(512, 256)
        
        self.up_1 = UpSample(512, 128)
        self.up_2 = UpSample(256, 64)
        self.up_3 = UpSample(128, 64)
        
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
    
    def pos_encoding(self, t, channels):
        # TODO : 位置编码的作用
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        if not self.cpu:
            inv_freq = inv_freq.cuda()
        
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.embed_time_dim)
        
        x1 = self.inc(x)
        x2 = self.sa_1(self.down_1(x1, t))
        x3 = self.sa_2(self.down_2(x2, t))
        x4 = self.sa_3(self.down_3(x3, t))
        
        x4 = self.bot_3(self.bot_2(self.bot_1(x4)))
        
        x = self.sa_4(self.up_1(x4, x3, t))
        x = self.sa_5(self.up_2(x, x2, t))
        x = self.sa_6(self.up_3(x, x1, t))
        
        return self.outc(x)


if __name__ == '__main__':
    model = Unet(3, 3, True, 256)
    
        
        