from torch import nn
import torch
from modules import MeanFieldGaussianWithNodeVariance, MeanFieldGaussian
import torch.nn.functional as F
from .vcl_bayesian_linear import VCLBayesianLinear
import utils
import math


class UCLBayesianLinear(VCLBayesianLinear):
    def __init__(self, input_size, output_size, lambda_logvar=-5.0, ratio=0.5, alpha=0.01, beta=0.03, gamma=0.0, previous_ucl_layer=None):
        super(UCLBayesianLinear, self).__init__(input_size, output_size, lambda_logvar)
        self.posterior_weights = MeanFieldGaussianWithNodeVariance(input_size, output_size, lambd_logvar=lambda_logvar)
        self.posterior_biases = MeanFieldGaussianWithNodeVariance(1, output_size, lambd_logvar=lambda_logvar)
        self.prev_post_weights = MeanFieldGaussianWithNodeVariance(input_size, output_size, lambd_logvar=lambda_logvar)
        self.prev_post_biases = MeanFieldGaussianWithNodeVariance(1, output_size, lambd_logvar=lambda_logvar)
        self.fan_in = input_size
        self.fan_out = output_size
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.previous_ucl_layer = previous_ucl_layer

    def posterior_kl_div(self):
        prev_layer_prev_post_weights_logvar = self.previous_ucl_layer.prev_post_weights.logvar if self.previous_ucl_layer else None
        prev_layer_prev_post_biases_logvar = self.previous_ucl_layer.prev_post_biases.logvar if self.previous_ucl_layer else None

        return self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights.mean, self.prev_post_weights.logvar, prev_layer_prev_post_weights_logvar) \
            + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases.mean, self.prev_post_biases.logvar, prev_layer_prev_post_biases_logvar)

    @property
    def std_init(self):
        return math.sqrt((2 / self.fan_in) * self.ratio)


    def _kl_div(self, mu_post, logvar_post, mu_prev_post, logvar_prev_post, prev_layer_logvar_prev_post):
        sigma_post = torch.exp(0.5 * logvar_post)
        sigma_prev_post = torch.exp(0.5 * logvar_prev_post)
        

        if prev_layer_logvar_prev_post is None:
            prev_layer_prev_post_strength = torch.zeros(1, self.fan_in).to(sigma_post.device)
        else: 
            sigma_prev_layer_prev_post = torch.exp(0.5 * prev_layer_logvar_prev_post)
            prev_layer_std_init = self.previous_ucl_layer.std_init
            prev_layer_prev_post_strength = prev_layer_std_init / sigma_prev_layer_prev_post
        
        prev_post_strength = self.std_init / sigma_prev_post

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
        