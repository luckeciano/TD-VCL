from torch import nn
import torch
import torch.nn.functional as F
from .vcl_bayesian_conv import VCLBayesianConv2D, MeanFieldGaussianConvBiases
import utils
import math
from torch.nn.modules.utils import _pair

class MeanFieldGaussianConvWeightsNodeVariance(nn.Module):
    def __init__(self, out_channels, in_channels, groups, kernel_size, lambda_logvar=-5.0):
        super(MeanFieldGaussianConvWeightsNodeVariance, self).__init__()
        self.mean = nn.Parameter(torch.normal(mean=0, std=0.1, size=((out_channels, in_channels//groups, *kernel_size))))
        self.logvar = nn.Parameter(lambda_logvar * torch.ones((out_channels, 1, 1, 1),
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
    

class UCLBayesianConv2D(VCLBayesianConv2D):
        def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, lambda_logvar=-15.0, ratio=0.5, alpha=0.01, beta=0.03, gamma=0.0, previous_ucl_layer=None):
            super(UCLBayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, 
                                                stride, padding, dilation, groups, lambda_logvar)
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
                        
            self.posterior_weights = MeanFieldGaussianConvWeightsNodeVariance(out_channels, in_channels, groups, kernel_size, lambda_logvar)
            self.posterior_biases = MeanFieldGaussianConvBiases(out_channels, lambda_logvar)
            self.prev_post_weights = MeanFieldGaussianConvWeightsNodeVariance(out_channels, in_channels, groups, kernel_size, lambda_logvar)
            self.prev_post_biases = MeanFieldGaussianConvBiases(out_channels, lambda_logvar)

            self.fan_in, self.fan_out = utils._calculate_fan_in_and_fan_out(self.posterior_weights.mean)
            self.ratio = ratio
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.previous_ucl_layer = previous_ucl_layer
                
        def posterior_kl_div(self):
            prev_layer_prev_post_weights_logvar = self.previous_ucl_layer.prev_post_weights.logvar if self.previous_ucl_layer else None

            return self._kl_div_weights(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights.mean, self.prev_post_weights.logvar, prev_layer_prev_post_weights_logvar) \
                + self._kl_div_biases(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases.mean, self.prev_post_biases.logvar)

        @property
        def std_init(self):
            return math.sqrt((2 / self.fan_out) * self.ratio)


        def _kl_div_weights(self, mu_post, logvar_post, mu_prev_post, logvar_prev_post, prev_layer_logvar_prev_post):
            sigma_post = torch.exp(0.5 * logvar_post)
            sigma_prev_post = torch.exp(0.5 * logvar_prev_post)
            

            if prev_layer_logvar_prev_post is None:
                prev_layer_prev_post_strength = torch.zeros(3, 1, 1, 1).to(sigma_post.device)
            else: 
                sigma_prev_layer_prev_post = torch.exp(0.5 * prev_layer_logvar_prev_post)
                prev_layer_std_init = self.previous_ucl_layer.std_init
                prev_layer_prev_post_strength = prev_layer_std_init / sigma_prev_layer_prev_post

            
            prev_post_strength = self.std_init / sigma_prev_post

            out_features, in_features, _, _ = mu_prev_post.shape
            prev_post_strength = prev_post_strength.expand(out_features,in_features,1,1)
            prev_layer_prev_post_strength = prev_layer_prev_post_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)

            # Terms
            # Term (4)
            l2_strength = torch.maximum(prev_post_strength, prev_layer_prev_post_strength)

            reg_matrix = l2_strength * (mu_post - mu_prev_post)
            mu_reg = (reg_matrix).norm(2)**2
            
            # Term (5)
            l1_mu_reg = (torch.div(mu_prev_post**2, sigma_prev_post**2) * (mu_post - mu_prev_post)).norm(1)
            l1_mu_reg = l1_mu_reg * (self.std_init ** 2)

            # Term (6)
            weight_sigma = (sigma_post**2 / sigma_prev_post**2)

            sigma_reg_sum = torch.sum(weight_sigma - torch.log(weight_sigma))
            sigma_normal_reg_sum = torch.sum(sigma_post**2 - torch.log(sigma_post**2))

            return self.alpha * (mu_reg/ 2) + self.gamma * l1_mu_reg + self.beta * (sigma_reg_sum + sigma_normal_reg_sum) / 2
        

        def _kl_div_biases(self, mu_post, logvar_post, mu_prev_post, logvar_prev_post):
            sigma_post = torch.exp(0.5 * logvar_post)
            sigma_prev_post = torch.exp(0.5 * logvar_prev_post)
            
            prev_post_strength = self.std_init / sigma_prev_post

            # Terms
            # Term (4)
            bias_strength = prev_post_strength

            reg_matrix = bias_strength * (mu_post - mu_prev_post)
            mu_reg = (reg_matrix).norm(2)**2
            
            # Term (5)
            l1_mu_reg = (torch.div(mu_prev_post**2, sigma_prev_post**2) * (mu_post - mu_prev_post)).norm(1)
            l1_mu_reg = l1_mu_reg * (self.std_init ** 2)

            # Term (6)
            weight_sigma = (sigma_post**2 / sigma_prev_post**2)

            sigma_reg_sum = torch.sum(weight_sigma - torch.log(weight_sigma))
            sigma_normal_reg_sum = torch.sum(sigma_post**2 - torch.log(sigma_post**2))

            return self.alpha * (mu_reg/ 2) + self.gamma * l1_mu_reg + self.beta * (sigma_reg_sum + sigma_normal_reg_sum) / 2

    
        