# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/12/01 10:39:07
# @FileName:   base_model_gan.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import os
import math
import torch
import torch.optim as optim

from source.loss import Loss
from source.net import Network
from source.metric import Metric
from source.dataset import DataSet
from torch.utils.data import DataLoader

class BaseGAN:
    def __init__(self, config, accelerator) -> None:
        self.config_train = config['train']
        self.config_val = config['val']
        self.resume_info = config['train']['resume']

        self.accelerator = accelerator
        self.num_nodes = self.conf_train['num_node']
        self.bacth_per_gpu = self.conf_train['batch_per_gpu']
        self.num_gpu_per_node = self.conf_train['num_gpu_per_node']

        self.val_freq = self.config_val['val_freq']
        self.save_freq_g = self.config_train['save_freq']['net_g']
        self.save_freq_d = self.config_train['save_freq']['net_d']
        self.show_iter = self.config_val['show_iter']
        self.print_freq = self.conf_train['print_freq']
        self.total_iter = self.conf_train['total_iters']

        self.loss_d = 0
        self.loss_g = 0
        self.cur_iter = 0

        # 初始化
        criterion = Loss(config)()
        dataset = DataSet(config)()

        if not dataset.get('test', False):
            self.val_sum = 0
        else:
            self.val_sum = dataset['test'].__len__()
        
        train_loader = DataLoader(dataset['train'],
                                  batch_size=self.bacth_per_gpu,
                                  shuffle=True,
                                  num_workers=self.conf_train['num_worker'])
        test_loader = DataLoader(dataset['test'], batch_size=1, shuffle=False, num_workers=0)

        net_g = Network(config)()
        net_d = Network(config)()