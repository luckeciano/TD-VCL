from torch import nn
import torch
from modules import MeanFieldGaussianWithNodeVariance, UCLBayesianLinear
import torch.nn.functional as F

class TDUCLBayesianLinear(UCLBayesianLinear):

    def __init__(self, input_size, output_size, n, lambd, lambda_logvar=-5.0, ratio=0.5, alpha=0.01, beta=0.03, gamma=0.0, previous_ucl_layer=None):
        super(TDUCLBayesianLinear, self).__init__(input_size, output_size, lambda_logvar, ratio, alpha, beta, gamma, previous_ucl_layer)
        self.input_size = input_size
        self.output_size = output_size
        self.n = n
        self.lambd = lambd
        self.posterior_weights = MeanFieldGaussianWithNodeVariance(input_size, output_size, lambd_logvar=lambda_logvar)
        self.posterior_biases = MeanFieldGaussianWithNodeVariance(1, output_size, lambd_logvar=lambda_logvar)

        self.prev_post_weights = nn.ModuleList()
        self.prev_post_biases = nn.ModuleList()

        for i in range(n):
            self.prev_post_weights.append(MeanFieldGaussianWithNodeVariance(input_size, output_size, lambd_logvar=lambda_logvar))
            self.prev_post_biases.append(MeanFieldGaussianWithNodeVariance(1, output_size, lambd_logvar=lambda_logvar))

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
        prev_layer_prev_post_biases_logvar = self.previous_ucl_layer.prev_post_biases[self.n - 1].logvar if self.previous_ucl_layer else None

        k = self.n - curr_timestep if curr_timestep < self.n else 1 # Decide the oldest posterior to consider
        discount = 1.0
        steps = 0.0
        kl_div = 0

        step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[self.n - 1].mean, self.prev_post_weights[self.n - 1].logvar, prev_layer_prev_post_weights_logvar) \
            + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[self.n - 1].mean, self.prev_post_biases[self.n - 1].logvar, prev_layer_prev_post_biases_logvar)
        steps += 1.0

        kl_div = kl_div + discount * step_kl
        
        for i in reversed(range(k, self.n - 1)):
            discount = self.lambd * discount
            step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[i].mean, self.prev_post_weights[i].logvar, prev_layer_prev_post_weights_logvar) \
                + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[i].mean, self.prev_post_biases[i].logvar, prev_layer_prev_post_biases_logvar)
            kl_div = kl_div + discount * step_kl
            steps += 1.0

        norm = (self.lambd - 1.0) / (self.lambd ** steps - 1.0)
        
        return kl_div * norm