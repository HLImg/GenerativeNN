# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/12/05 15:40:08
# @File    :   ddpm.py
# @Contact   :   lianghao@whu.edu.cn

import torch

from tqdm import tqdm
from source.model.base_model import BaseModel
from source.utils.image.transpose import tensor2img

class DDPModel(BaseModel):
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        
        ddpm = config['model']['ddpm']
        
        self.noise_steps = ddpm['noise_steps']
        self.beta_start = ddpm['beta_start']
        self.beta_end = ddpm['beta_end']
        self.img_size = ddpm['img_size']
        
        self.beta = self.to_device(self.prepare_noise_schedule())
        self.alpha = 1 - self.beta
        # \bar\alpha_t = \prod_{i=1}^t alpha_i
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def to_device(self, x):
        return self.accelerator.prepare(x)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))
    
    def noise_images(self, x_0, t):
        """forward diffusion process
        x_t = \sqrt{\bar\alpha_t}\cdot x_0 + \sqrt{1 - \bar\alpha_t} \cdot \epsilon
            \text{ where } \epsilon \sim \mathcal N(0, \mathbf I)
        Args:
            x (image): x_0
            t (i-th): timesteps
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        epsilon = torch.randn_like(x_0)
        return {'x':sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon,
                'epsilon': epsilon}
    
    def sample(self, n):
        """reverse diffusion process
        x_t \sim N(0, 1)
        for t = T, ..., 1 do
            z \sim N(0, 1) if t > 1 else z = 0
            x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar\alpha_t}}\cdot \epsilon_\theta(x_t, t)) + \sigma_t\cdot z
        end for
        
        In the paper of DDPM, the setting of  $\sigma_t^2$ is $\beta_t$. 
        Args:
            n (int): 
        """
        self.net_g.eval()
        with torch.no_grad():
            x_t = torch.randn((n, 3, self.img_size, self.img_size)).cuda()
            for t in tqdm(reversed(range(1, self.noise_steps)), position=0):
                embed_t = (torch.ones(n) * t).long().cuda()
                predicted_epsilon = self.net_g(x, embed_t)
                alpha_t = self.alpha[t][:, None, None, None]
                alpha_bar_t = self.alpha_bar[t]
                sigma_t = torch.sqrt(self.beta[t][:, None, None, None])
                
                if t > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha_t) * (x - predicted_epsilon * (
                    (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t))) + z * sigma_t
        
        return x
    
    def __feed__(self, data):
        self.optimizer.zero_grad()
        image = data['lq']
        t = self.sample_timesteps(image.shape[0]).cuda()
        x_t, epsilon = self.noise_images(image, t)
        predicted_epsilon = self.net_g(x_t, t)
        loss = self.criterion(epsilon, predicted_epsilon)
        self.loss = loss.item()

        # ================================== #
        self.accelerator.backward(loss)
        # ================================== #
        self.optimizer.step()
        self.scheduler.step()
    
    def __eval__(self, data):
        return {}