# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/2 18:53
# @File    :   __init__.py.py
# @Email   :   lianghao@whu.edu.cn

from .denoise import *
from .generative import *
from .common.pair_data import PairDataset

datasets = {
    'denoise': {
        'pair_lmdb': PairDataset 
    },
    'generative':{
        'vae_celeba': CelebaDataSet
    }
}

class DataSet:
    def __init__(self, config):
        self.info = config["dataset"]
        self.task = self.info["task"]
        self.name = self.info["name"]
        self.params = self.info["param"]

    def __call__(self, *args, **kwargs):
        if self.task not in datasets:
            raise ValueError("the name of dataset is not exits")
        
        train_dataset, test_dataset = (None, None)
        
        if 'train' in self.params.keys():
            train_dataset = datasets[self.task][self.name](**self.params["train"])
        
        if 'test' in self.params.keys():
            test_dataset = datasets[self.task][self.name](**self.params["test"])
            
        return {"train": train_dataset, "test": test_dataset}
        