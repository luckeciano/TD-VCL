import numpy as np
import torch
import torch.nn as nn

from .td_ucl_bayesian_linear import TDUCLBayesianLinear
from .td_ucl_bayesian_conv import TDUCLBayesianConv2D


class TDUCLBayesianAlexNet(nn.Module):
    def __init__(self, inputsize, n, lambd, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_mlp=-15.0, ratio=0.5, alpha=1.0, beta=0.03, gamma=1.0):
        super(TDUCLBayesianAlexNet, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = TDUCLBayesianConv2D(ncha, 64, kernel_size=3, n = n, lambd=lambd, stride=1, padding=1, lambda_logvar=lambda_logvar, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=None)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = TDUCLBayesianConv2D(64, 192, kernel_size=3, n = n, lambd=lambd, padding=1, lambda_logvar=lambda_logvar, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.conv1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = TDUCLBayesianConv2D(192, 384, kernel_size=3, n = n, lambd=lambd, padding=1, lambda_logvar=lambda_logvar, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.conv2)
        
        self.conv4 = TDUCLBayesianConv2D(384, 256, kernel_size=3, n = n, lambd=lambd, padding=1, lambda_logvar=lambda_logvar, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.conv3)
        
        self.conv5 = TDUCLBayesianConv2D(256, 256, kernel_size=3, n = n, lambd=lambd, padding=1, lambda_logvar=lambda_logvar, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.conv4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = TDUCLBayesianLinear(256 * s * s, 4096, n = n, lambd=lambd, lambda_logvar=lambda_logvar_mlp, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.conv5)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = TDUCLBayesianLinear(4096, 4096, n = n, lambd=lambd, lambda_logvar=lambda_logvar_mlp, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, previous_ucl_layer=self.fc1)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))

        self.bayesian_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]
            
        self.relu = nn.ReLU()
        self.set_task(0)
        
    def set_task(self, id):
        self.current_task = id

    def kl_div(self, curr_timestep):
        kl_div = 0
        for layer in self.bayesian_layers:
                kl_div += layer.posterior_kl_div(curr_timestep)
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.bayesian_layers:
                layer.update_posterior()

    def _backbone(self, x, sample=True):
        x = self.relu(self.conv1(x, sample))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x, sample))
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x, sample))
        
        x = self.relu(self.conv4(x, sample))
        
        x = self.relu(self.conv5(x, sample))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu(self.fc1(x, sample)))
        x = self.drop2(self.relu(self.fc2(x, sample)))
        return x
         
        
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        x = self.last[self.current_task](x)
        
        return x
    
class MultiHeadTDUCLBayesianAlexNet(TDUCLBayesianAlexNet):
    def __init__(self, inputsize, n, lambd, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_mlp=-15.0, ratio=0.5, alpha=1.0, beta=0.03, gamma=1.0):
        super(MultiHeadTDUCLBayesianAlexNet, self).__init__(inputsize, n, lambd, num_heads, num_classes, lambda_logvar, lambda_logvar_mlp, ratio, alpha, beta, gamma)
    
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        outputs = torch.cat([head(x).unsqueeze(1) for head in self.last], axis=1)
        return outputs