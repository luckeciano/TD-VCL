import torch.nn as nn
import math
from torch.nn.modules.utils import _pair
import utils
from .ucl_bayesian_conv import MeanFieldGaussianConvWeightsNodeVariance, MeanFieldGaussianConvBiases, UCLBayesianConv2D

class TDUCLBayesianConv2D(UCLBayesianConv2D):
        def __init__(self, in_channels, out_channels, kernel_size, n, lambd,
                 stride=1, padding=0, dilation=1, groups=1, lambda_logvar=-15.0, ratio=0.5, alpha=0.01, beta=0.03, gamma=0.0, previous_ucl_layer=None):
            super(TDUCLBayesianConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                                stride, padding, dilation, groups, lambda_logvar, ratio, alpha, beta, gamma, previous_ucl_layer)
            self.lambd = lambd
            self.n = n
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
                        
            self.posterior_weights = MeanFieldGaussianConvWeightsNodeVariance(out_channels, in_channels, groups, kernel_size, lambda_logvar)
            self.posterior_biases = MeanFieldGaussianConvBiases(out_channels, lambda_logvar)

            self.prev_post_weights = nn.ModuleList()
            self.prev_post_biases = nn.ModuleList()

            for i in range(n):
                self.prev_post_weights.append(MeanFieldGaussianConvWeightsNodeVariance(out_channels, in_channels, groups, kernel_size, lambda_logvar))
                self.prev_post_biases.append(MeanFieldGaussianConvBiases(out_channels, lambda_logvar))

            self.fan_in, self.fan_out = utils._calculate_fan_in_and_fan_out(self.posterior_weights.mean)
            
        def update_posterior(self):
            for i in range(1, self.n - 1):
                self.prev_post_weights[i - 1].mean.data.copy_(self.prev_post_weights[i].mean)
                self.prev_post_weights[i - 1].logvar.data.copy_(self.prev_post_weights[i].logvar)
                self.prev_post_weights[i - 1].mean.requires_grad = False
                self.prev_post_weights[i - 1].logvar.requires_grad = False

                self.prev_post_biases[i - 1].mean.data.copy_(self.prev_post_biases[i].mean)
                self.prev_post_biases[i - 1].logvar.data.copy_(self.prev_post_biases[i].logvar)
                self.prev_post_biases[i - 1].mean.requires_grad = False
                self.prev_post_biases[i - 1].logvar.requires_grad = False

            self.prev_post_weights[self.n - 1].mean.data.copy_(self.posterior_weights.mean)
            self.prev_post_weights[self.n - 1].logvar.data.copy_(self.posterior_weights.logvar)
            self.prev_post_weights[self.n - 1].mean.requires_grad = False
            self.prev_post_weights[self.n - 1].logvar.requires_grad = False

            self.prev_post_biases[self.n - 1].mean.data.copy_(self.posterior_biases.mean)
            self.prev_post_biases[self.n - 1].logvar.data.copy_(self.posterior_biases.logvar)
            self.prev_post_biases[self.n - 1].mean.requires_grad = False
            self.prev_post_biases[self.n - 1].logvar.requires_grad = False
        
        def posterior_kl_div(self, curr_timestep):
            prev_layer_prev_post_weights_logvar = self.previous_ucl_layer.prev_post_weights[self.n - 1].logvar if self.previous_ucl_layer else None

            k = self.n - curr_timestep if curr_timestep < self.n else 1 # Decide the oldest posterior to consider
            discount = 1.0
            steps = 0.0
            kl_div = 0

            step_kl = self._kl_div_weights(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[self.n - 1].mean, self.prev_post_weights[self.n - 1].logvar, prev_layer_prev_post_weights_logvar) \
                + self._kl_div_biases(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[self.n - 1].mean, self.prev_post_biases[self.n - 1].logvar)
            steps += 1.0

            kl_div = kl_div + discount * step_kl
            
            for i in reversed(range(k, self.n - 1)):
                discount = self.lambd * discount
                step_kl = self._kl_div_weights(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[i].mean, self.prev_post_weights[i].logvar, prev_layer_prev_post_weights_logvar) \
                    + self._kl_div_biases(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[i].mean, self.prev_post_biases[i].logvar)
                kl_div = kl_div + discount * step_kl
                steps += 1.0

            norm = (self.lambd - 1.0) / (self.lambd ** steps - 1.0)
            
            return kl_div * norm