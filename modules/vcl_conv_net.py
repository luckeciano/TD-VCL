import numpy as np
import torch
import torch.nn as nn
from .vcl_bayesian_conv import BayesianConv2D, VCLBayesianConv2D
from .vcl_bayesian_batch_norm import VCLBayesianBatchNorm2D
from .vcl_bayesian_linear import VCLBayesianLinear, BayesianLinear

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class VCLBayesianConvNet(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10, lambda_logvar=-15.0):
        super().__init__()
        
        ncha, size, _= inputsize
        
        self.conv1 = VCLBayesianConv2D(ncha,32,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = VCLBayesianConv2D(32,32,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = VCLBayesianConv2D(32,64,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = VCLBayesianConv2D(64,64,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = VCLBayesianConv2D(64,128,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = VCLBayesianConv2D(128,128,kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = VCLBayesianLinear(input_size = s*s*128, output_size=256, lambda_logvar=lambda_logvar)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for _ in range(num_heads):
            self.last.append(torch.nn.Linear(256,num_classes))
        self.relu = torch.nn.ReLU()

        self.bayesian_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1]

    def forward(self, x, sample=True):
        h=self.relu(self.conv1(x,sample))
        h=self.relu(self.conv2(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h,sample))
        h=self.relu(self.conv4(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h,sample))
        h=self.relu(self.conv6(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h,sample)))
        y = self.last[self.current_task](h)
        
        return y

    def kl_div(self):
        kl_div = 0
        for layer in self.bayesian_layers:
                kl_div += layer.posterior_kl_div()
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.bayesian_layers:
                layer.update_posterior()


class VCLBayesianAlexNet(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_mlp=-15.0):
        super(VCLBayesianAlexNet, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = VCLBayesianConv2D(ncha, 64, kernel_size=3, stride=1, padding=1, lambda_logvar=lambda_logvar)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = VCLBayesianConv2D(64, 192, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = VCLBayesianConv2D(192, 384, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv4 = VCLBayesianConv2D(384, 256, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv5 = VCLBayesianConv2D(256, 256, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = VCLBayesianLinear(256 * s * s, 4096, lambda_logvar=lambda_logvar_mlp)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = VCLBayesianLinear(4096, 4096, lambda_logvar=lambda_logvar_mlp)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))

        self.bayesian_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]
            
        self.relu = nn.ReLU()
        self.set_task(0)
        
    def set_task(self, id):
        self.current_task = id

    def kl_div(self):
        kl_div = 0
        for layer in self.bayesian_layers:
                kl_div += layer.posterior_kl_div()
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
    
class VCLBayesianAlexNetV2(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_batchnorm=-5.0, lambda_logvar_mlp=-15.0):
        super(VCLBayesianAlexNetV2, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = VCLBayesianConv2D(ncha, 64, kernel_size=3, stride=1, padding=1, lambda_logvar=lambda_logvar)
        self.conv1_bn = VCLBayesianBatchNorm2D(64, lambda_logvar_batchnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = VCLBayesianConv2D(64, 192, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.conv2_bn = VCLBayesianBatchNorm2D(192, lambda_logvar_batchnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = VCLBayesianConv2D(192, 384, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.conv3_bn = VCLBayesianBatchNorm2D(384, lambda_logvar_batchnorm)

        self.conv4 = VCLBayesianConv2D(384, 256, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.conv4_bn = VCLBayesianBatchNorm2D(256, lambda_logvar_batchnorm)

        self.conv5 = VCLBayesianConv2D(256, 256, kernel_size=3, padding=1, lambda_logvar=lambda_logvar)
        self.conv5_bn = VCLBayesianBatchNorm2D(256, lambda_logvar_batchnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = VCLBayesianLinear(256 * s * s, 4096, lambda_logvar=lambda_logvar_mlp)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = VCLBayesianLinear(4096, 4096, lambda_logvar=lambda_logvar_mlp)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))

        self.bayesian_layers = [self.conv1, self.conv1_bn, self.conv2, self.conv2_bn, self.conv3, self.conv3_bn, self.conv4, self.conv4_bn, self.conv5, self.conv5_bn, self.fc1, self.fc2]
            
        self.relu = nn.ReLU()
        self.set_task(0)
        
    def set_task(self, id):
        self.current_task = id

    def kl_div(self):
        kl_div = 0
        for layer in self.bayesian_layers:
                kl_div += layer.posterior_kl_div()
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
    
class MultiHeadVCLBayesianAlexNetV2(VCLBayesianAlexNetV2):
    def __init__(self, inputsize, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_batchnorm=-5.0, lambda_logvar_mlp=-15.0):
        super(MultiHeadVCLBayesianAlexNetV2, self).__init__(inputsize, num_heads, num_classes, lambda_logvar, lambda_logvar_batchnorm, lambda_logvar_mlp)
    
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        outputs = torch.cat([head(x).unsqueeze(1) for head in self.last], axis=1)
        return outputs
    
class VCLBayesianAlexNet64(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=20, lambda_logvar=-15.0, lambda_logvar_mlp=-15.0):
        super(VCLBayesianAlexNet64, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = VCLBayesianConv2D(ncha, 64, kernel_size=5, stride=1, padding=2, lambda_logvar=lambda_logvar)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = VCLBayesianConv2D(64, 192, kernel_size=5, stride=1, padding=2, lambda_logvar=lambda_logvar)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = VCLBayesianConv2D(192, 384, kernel_size=3, stride=1, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv4 = VCLBayesianConv2D(384, 256, kernel_size=3, stride=1, padding=1, lambda_logvar=lambda_logvar)
        
        self.conv5 = VCLBayesianConv2D(256, 256, kernel_size=3, stride=1, padding=1, lambda_logvar=lambda_logvar)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # Size is reduced by half at each of the three pooling layers
        
        self.fc1 = VCLBayesianLinear(256 * s * s, 1024, lambda_logvar=lambda_logvar_mlp)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = VCLBayesianLinear(1024, 1024, lambda_logvar=lambda_logvar_mlp)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(1024, num_classes))

        self.bayesian_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]
            
        self.relu = nn.ReLU()
        self.set_task(0)
        
    def set_task(self, id):
        self.current_task = id

    def kl_div(self):
        kl_div = 0
        for layer in self.bayesian_layers:
            kl_div += layer.posterior_kl_div()
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.bayesian_layers:
            layer.update_posterior()

    def _backbone(self, x, sample):
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
    
class MultiHeadVCLBayesianAlexNet64(VCLBayesianAlexNet64):
    def __init__(self, inputsize, num_heads=1, num_classes=10, lambda_logvar=-15.0, lambda_logvar_batchnorm=-5.0, lambda_logvar_mlp=-15.0):
        super(MultiHeadVCLBayesianAlexNet64, self).__init__(inputsize, num_heads, num_classes, lambda_logvar, lambda_logvar_batchnorm, lambda_logvar_mlp)
    
    def forward(self, x, sample=True):
        x = self._backbone(x, sample)
        outputs = torch.cat([head(x).unsqueeze(1) for head in self.last], axis=1)
        return outputs