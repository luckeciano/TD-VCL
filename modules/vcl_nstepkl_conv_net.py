import numpy as np
import torch
import torch.nn as nn
from .vcl_nstepkl_bayesian_conv import NStepKLVCLBayesianConv2D
from .vcl_nstepkl import NStepKLVCLBayesianLinear
from .vcl_nstepkl_bayesian_batch_norm import NStepKLVCLBayesianBatchNorm2D


class NStepKLVCLBayesianAlexNet(nn.Module):
    def __init__(self, inputsize, n, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_mlp=-15.0):
        super(NStepKLVCLBayesianAlexNet, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = NStepKLVCLBayesianConv2D(ncha, 64, kernel_size=3, n = n, stride=1, padding=1, lambda_logvar=lambda_logvar)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = NStepKLVCLBayesianConv2D(64, 192, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = NStepKLVCLBayesianConv2D(192, 384, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv4 = NStepKLVCLBayesianConv2D(384, 256, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv5 = NStepKLVCLBayesianConv2D(256, 256, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = NStepKLVCLBayesianLinear(256 * s * s, 4096, n = n, lambda_logvar=lambda_logvar_mlp)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = NStepKLVCLBayesianLinear(4096, 4096, n = n, lambda_logvar=lambda_logvar_mlp)
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
        
    def forward(self, x, sample=True):
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
        # x = self.fc3(x)

        x = self.last[self.current_task](x)
        
        return x
    
class NStepKLVCLBayesianAlexNetV2(nn.Module):
    def __init__(self, inputsize, n, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_batchnorm=-5.0, lambda_logvar_mlp=-15.0):
        super(NStepKLVCLBayesianAlexNetV2, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = NStepKLVCLBayesianConv2D(ncha, 64, kernel_size=3, n = n, stride=1, padding=1, lambda_logvar=lambda_logvar)
        self.conv1_bn = NStepKLVCLBayesianBatchNorm2D(64, n, lambda_logvar_batchnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = NStepKLVCLBayesianConv2D(64, 192, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.conv2_bn = NStepKLVCLBayesianBatchNorm2D(192, n, lambda_logvar_batchnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = NStepKLVCLBayesianConv2D(192, 384, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.conv3_bn = NStepKLVCLBayesianBatchNorm2D(384, n, lambda_logvar_batchnorm)

        self.conv4 = NStepKLVCLBayesianConv2D(384, 256, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.conv4_bn = NStepKLVCLBayesianBatchNorm2D(256, n, lambda_logvar_batchnorm)

        self.conv5 = NStepKLVCLBayesianConv2D(256, 256, kernel_size=3, n = n, padding=1, lambda_logvar=lambda_logvar)
        self.conv5_bn = NStepKLVCLBayesianBatchNorm2D(256, n, lambda_logvar_batchnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = NStepKLVCLBayesianLinear(256 * s * s, 4096, n = n, lambda_logvar=lambda_logvar_mlp)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = NStepKLVCLBayesianLinear(4096, 4096, n = n, lambda_logvar=lambda_logvar_mlp)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))

        self.bayesian_layers = [self.conv1, self.conv1_bn, self.conv2, self.conv2_bn, self.conv3, self.conv3_bn, self.conv4, self.conv4_bn, self.conv5, self.conv5_bn, self.fc1, self.fc2]
            
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
    
    def _backbone(self, x, sample):
        x = self.relu(self.conv1_bn(self.conv1(x, sample), sample))
        x = self.pool1(x)
        
        x = self.relu(self.conv2_bn(self.conv2(x, sample), sample))
        x = self.pool2(x)
        
        x = self.relu(self.conv3_bn(self.conv3(x, sample), sample))
        
        x = self.relu(self.conv4_bn(self.conv4(x, sample), sample))
        
        x = self.relu(self.conv5_bn(self.conv5(x, sample), sample))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu(self.fc1(x, sample)))
        x = self.drop2(self.relu(self.fc2(x, sample)))
        return x
        
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        x = self.last[self.current_task](x)
        
        return x
    
class MultiHeadNStepKLVCLBayesianAlexNetV2(NStepKLVCLBayesianAlexNetV2):
    def __init__(self, inputsize, n, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_batchnorm=-5.0, lambda_logvar_mlp=-15.0):
        super(MultiHeadNStepKLVCLBayesianAlexNetV2, self).__init__(inputsize, n, num_heads, num_classes, lambda_logvar, lambda_logvar_batchnorm, lambda_logvar_mlp)
    
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        outputs = torch.cat([head(x).unsqueeze(1) for head in self.last], axis=1)
        return outputs