import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair

class MeanFieldGaussianConvWeights(nn.Module):
    def __init__(self, out_channels, in_channels, groups, kernel_size, lambda_logvar=-5.0):
        super(MeanFieldGaussianConvWeights, self).__init__()
        self.mean = nn.Parameter(torch.normal(mean=0, std=0.1, size=((out_channels, in_channels//groups, *kernel_size))))
        self.logvar = nn.Parameter(lambda_logvar * torch.ones((out_channels, in_channels//groups, *kernel_size),
                                        dtype=torch.float32))
        self._initialize_weights() 
        
    def _initialize_weights(self): 
        nn.init.kaiming_normal_(self.mean, mode='fan_out', nonlinearity='relu')
    
    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)

            z = z + var * eps
        return z
    
class MeanFieldGaussianConvBiases(nn.Module):
    def __init__(self, out_channels, lambda_logvar=-5.0):
        super(MeanFieldGaussianConvBiases, self).__init__()
        self.mean = nn.Parameter(torch.zeros(size=((out_channels, ))))
        self.logvar = nn.Parameter(lambda_logvar * torch.ones(out_channels,
                                    dtype=torch.float32))

    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)

            z = z + var * eps
        return z

class _BayesianConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride,padding, dilation, transposed, output_padding, groups, lambda_logvar):
        super(_BayesianConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        self.posterior_weights = MeanFieldGaussianConvWeights(out_channels, in_channels, groups, kernel_size, lambda_logvar)
        self.posterior_biases = MeanFieldGaussianConvBiases(out_channels, lambda_logvar)

        
class BayesianConv2D(_BayesianConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, lambda_logvar=-15.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, 
                                             stride, padding, dilation, False, _pair(0), groups, lambda_logvar)
        
    def forward(self, input, sample = False):
        if sample:
            weight = self.posterior_weights(sample)
            bias = self.posterior_biases(sample)
            
        else:
            weight = self.posterior_weights.mean
            bias = self.posterior_biases.mean
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
class VCLBayesianConv2D(BayesianConv2D):
        def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, lambda_logvar=-15.0):
            super(VCLBayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, 
                                                stride, padding, dilation, groups, lambda_logvar)
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            
            self.prev_post_weights = MeanFieldGaussianConvWeights(out_channels, in_channels, groups, kernel_size, lambda_logvar)
            self.prev_post_biases = MeanFieldGaussianConvBiases(out_channels, lambda_logvar)
            
        def update_posterior(self):
            self.prev_post_weights.mean.data.copy_(self.posterior_weights.mean)
            self.prev_post_weights.logvar.data.copy_(self.posterior_weights.logvar)
            self.prev_post_weights.mean.requires_grad = False
            self.prev_post_weights.logvar.requires_grad = False

            self.prev_post_biases.mean.data.copy_(self.posterior_biases.mean)
            self.prev_post_biases.logvar.data.copy_(self.posterior_biases.logvar)
            self.prev_post_biases.mean.requires_grad = False
            self.prev_post_biases.logvar.requires_grad = False

        def posterior_kl_div(self):
            return self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights.mean, self.prev_post_weights.logvar) \
                + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases.mean, self.prev_post_biases.logvar)      

        def _kl_div(self, mu_post, logvar_post, mu_prev_post, logvar_prev_post):
            return - 0.5 * torch.sum( 1 + logvar_post - logvar_prev_post - (logvar_post.exp() / logvar_prev_post.exp()) - ((mu_post - mu_prev_post).pow(2) / logvar_prev_post.exp()) )
        

