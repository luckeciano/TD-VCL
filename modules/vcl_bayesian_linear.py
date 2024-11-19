from torch import nn
import torch
from modules import MeanFieldGaussian
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, input_size, output_size, lambda_logvar=-15.0):
        super(BayesianLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.posterior_weights = MeanFieldGaussian(N = input_size * output_size, lambd_logvar=lambda_logvar)
        self.posterior_biases = MeanFieldGaussian(N = output_size, lambd_logvar=lambda_logvar)

    def forward(self, x, sample):
        W = self.posterior_weights(sample).view(self.output_size, self.input_size)
        b = self.posterior_biases(sample).view(self.output_size)
        x = F.linear(x, W, b)
        return x


class VCLBayesianLinear(BayesianLinear):
    def __init__(self, input_size, output_size, lambda_logvar=-15.0):
        super(VCLBayesianLinear, self).__init__(input_size, output_size, lambda_logvar)
        self.prev_post_weights = MeanFieldGaussian(N = input_size * output_size, lambd_logvar=lambda_logvar)
        self.prev_post_biases = MeanFieldGaussian(N = output_size, lambd_logvar=lambda_logvar)

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
        