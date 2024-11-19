import torch
import torch.nn as nn
from .vcl_bayesian_batch_norm import BayesianBatchNorm2D, MeanFieldGaussianBatchNormWeights, MeanFieldGaussianBatchNormBiases
    

class NStepKLVCLBayesianBatchNorm2D(BayesianBatchNorm2D):
    def __init__(self, num_features, n, lambda_logvar=-5.0):
        super(NStepKLVCLBayesianBatchNorm2D, self).__init__(num_features, lambda_logvar)

        self.n = n
        self.prev_post_weights = nn.ModuleList()
        self.prev_post_biases = nn.ModuleList()

        for i in range(n):
            self.prev_post_weights.append(MeanFieldGaussianBatchNormWeights(num_features, lambda_logvar))
            self.prev_post_biases.append(MeanFieldGaussianBatchNormBiases(num_features, lambda_logvar))
        
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
    