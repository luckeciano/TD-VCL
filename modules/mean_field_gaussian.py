from torch import nn
import torch

class MeanFieldGaussian(nn.Module):
    def __init__(self, N, lambd_logvar):
        super(MeanFieldGaussian, self).__init__()
        self.mean = nn.Parameter(torch.normal(mean=0, std=0.1, size=(N,)))
        self.logvar = nn.Parameter(lambd_logvar * torch.ones(N))

    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)

            z = z + var * eps
        return z
    
class MeanFieldGaussianWithNodeVariance(nn.Module):
    def __init__(self, in_features, out_features, lambd_logvar):
        super(MeanFieldGaussianWithNodeVariance, self).__init__()
        self.mean = nn.Parameter(torch.normal(mean=0, std=0.1, size=(out_features, in_features)))
        self.logvar = nn.Parameter(lambd_logvar * torch.ones(out_features, 1))

    def forward(self, sample=True):
        z = self.mean
        if sample:
            var = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(var)

            z = z + var * eps
        return z
