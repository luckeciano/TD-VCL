import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanFieldGaussianBatchNormWeights(nn.Module):
    def __init__(self, num_features, lambda_logvar=-5.0):
        super(MeanFieldGaussianBatchNormWeights, self).__init__()
        self.mean = nn.Parameter(torch.normal(mean=0, std=0.1, size=(num_features,)))
        self.logvar = nn.Parameter(lambda_logvar * torch.ones(num_features, dtype=torch.float32))
        # self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.mean, mode='fan_out', nonlinearity='relu')
    
    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)
            z = z + var * eps
        return z

class MeanFieldGaussianBatchNormBiases(nn.Module):
    def __init__(self, num_features, lambda_logvar=-5.0):
        super(MeanFieldGaussianBatchNormBiases, self).__init__()
        self.mean = nn.Parameter(torch.zeros(size=(num_features,)))
        self.logvar = nn.Parameter(lambda_logvar * torch.ones(num_features, dtype=torch.float32))
    
    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)
            z = z + var * eps
        return z

class BayesianBatchNorm2D(nn.Module):
    def __init__(self, num_features, lambda_logvar=-5.0):
        super(BayesianBatchNorm2D, self).__init__()
        self.num_features = num_features
        self.posterior_weights = MeanFieldGaussianBatchNormWeights(num_features, lambda_logvar)
        self.posterior_biases = MeanFieldGaussianBatchNormBiases(num_features, lambda_logvar)
        self.register_buffer('running_mean', torch.zeros(self.num_features).normal_(0., 1.))
        self.register_buffer('running_var', torch.zeros(self.num_features).normal_(0., 1.))
        self.momentum = 0.1
        self.eps = 1e-5
    
    def forward(self, input, sample=True):
        if sample:
            weight = self.posterior_weights(sample)
            bias = self.posterior_biases(sample)
        else:
            weight = self.posterior_weights.mean
            bias = self.posterior_biases.mean
        
        return F.batch_norm(input, self.running_mean, self.running_var, weight, bias, self.training, self.momentum, self.eps)
    

class VCLBayesianBatchNorm2D(BayesianBatchNorm2D):
    def __init__(self, num_features, lambda_logvar=-5.0):
        super(VCLBayesianBatchNorm2D, self).__init__(num_features, lambda_logvar)

        self.prev_post_weights = MeanFieldGaussianBatchNormWeights(num_features, lambda_logvar)
        self.prev_post_biases = MeanFieldGaussianBatchNormBiases(num_features, lambda_logvar)
        
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
        
    
    
    