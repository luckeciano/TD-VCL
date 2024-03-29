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
