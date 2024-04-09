from torch import nn
import torch
from modules import MeanFieldGaussian
import torch.nn.functional as F

class NStepKLVCLBayesianLinear(nn.Module):

    def __init__(self, input_size, output_size, n, lambda_logvar=-5.0):
        super(NStepKLVCLBayesianLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n = n
        self.posterior_weights = MeanFieldGaussian(N = input_size * output_size, lambd_logvar=lambda_logvar)
        self.posterior_biases = MeanFieldGaussian(N = output_size, lambd_logvar=lambda_logvar)

        self.prev_post_weights = nn.ModuleList()
        self.prev_post_biases = nn.ModuleList()

        for i in range(n):
            self.prev_post_weights.append(MeanFieldGaussian(N = input_size * output_size, lambd_logvar=lambda_logvar))
            self.prev_post_biases.append(MeanFieldGaussian(N = output_size, lambd_logvar=lambda_logvar))

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
        k = self.n - curr_timestep if curr_timestep < self.n else 1 # Decide the oldest posterior to consider
        kl_div = 0
        num_kls = 0
        for i in range(k, self.n - 1):
            kl_div = kl_div + self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[i].mean, self.prev_post_weights[i].logvar) \
                + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[i].mean, self.prev_post_biases[i].logvar)
            num_kls += 1.0

        kl_div = kl_div + self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[self.n - 1].mean, self.prev_post_weights[self.n - 1].logvar) \
            + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[self.n - 1].mean, self.prev_post_biases[self.n - 1].logvar)   
        num_kls += 1.0   
        
        return kl_div / num_kls

    def _kl_div(self, mu_post, logvar_post, mu_prev_post, logvar_prev_post):
        return - 0.5 * torch.sum( 1 + logvar_post - logvar_prev_post - (logvar_post.exp() / logvar_prev_post.exp()) - ((mu_post - mu_prev_post).pow(2) / logvar_prev_post.exp()) )

    def forward(self, x, sample):
        W = self.posterior_weights(sample).view(self.output_size, self.input_size)
        b = self.posterior_biases(sample).view(self.output_size)
        x = F.linear(x, W, b)
        return x
        