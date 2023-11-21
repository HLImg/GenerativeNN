# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/18 15:20:19
# @FileName:  vae.py
# @Contact :  lianghao@whu.edu.cn

import torch
from source.model.base_model import BaseModel
from source.utils.image.transpose import tensor2img

class VAEModel(BaseModel):
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
    
    def __feed__(self, data):
        self.optimizer.zero_grad()
        y_hat, mean, logvar = self.net_g(data['lq'])
        loss = self.criterion(y_hat, data['lq'], mean, logvar)
        self.loss = loss.item() 
        
        # ========================================= #
        self.accelerator.backward(loss)
        # ========================================= #
        self.optimizer.step()
        self.scheduler.step()
    
    def __eval__(self, data):
        return {}