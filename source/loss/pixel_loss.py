# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:20
# @File    :   pixel_loss.py
# @Email   :   lianghao@whu.edu.cn
# @Thanks  :   BasicSR && NAFNet

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.weight * F.l1_loss(pred, target, reduction=self.reduction)

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.weight * F.mse_loss(pred, target, reduction=self.reduction)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class VAELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', kl_weight=0.00025):
        super(VAELoss, self).__init__()
        
        self.weight = loss_weight
        self.kl_weight = kl_weight
        self.reduction = reduction
        
        self.recon_loss = nn.MSELoss(reduction=reduction)
        self.kl_loss = lambda mean, logvar: torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    
    def forward(self, pred, target, mean, logvar):
        loss = self.recon_loss(pred, target) + self.kl_loss(mean, logvar) * self.kl_weight
        return self.weight * loss
    
