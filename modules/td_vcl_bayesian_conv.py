import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair
from .vcl_nstepkl_bayesian_conv import NStepKLVCLBayesianConv2D

class TDVCLBayesianConv2D(NStepKLVCLBayesianConv2D):
        def __init__(self, in_channels, out_channels, kernel_size, n, lambd,
                 stride=1, padding=0, dilation=1, groups=1, lambda_logvar=-15.0):
            super(TDVCLBayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, n, 
                                                stride, padding, dilation, groups, lambda_logvar)
            self.lambd = lambd
            
        def posterior_kl_div(self, curr_timestep):
            k = self.n - curr_timestep if curr_timestep < self.n else 1 # Decide the oldest posterior to consider
            discount = 1.0
            steps = 0.0
            kl_div = 0

            step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[self.n - 1].mean, self.prev_post_weights[self.n - 1].logvar) \
                + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[self.n - 1].mean, self.prev_post_biases[self.n - 1].logvar)
            steps += 1.0

            kl_div = kl_div + discount * step_kl
            
            for i in reversed(range(k, self.n - 1)):
                discount = self.lambd * discount
                step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[i].mean, self.prev_post_weights[i].logvar) \
                    + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[i].mean, self.prev_post_biases[i].logvar)
                kl_div = kl_div + discount * step_kl
                steps += 1.0

            norm = (self.lambd - 1.0) / (self.lambd ** steps - 1.0)
            
            return kl_div * norm