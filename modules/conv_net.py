import torch
import torch.nn as nn
import numpy as np

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class ConvNet(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10):
        super().__init__()
        
        ncha, size, _= inputsize
        
        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for _ in range(num_heads):
            self.last.append(torch.nn.Linear(256,num_classes))
        self.relu = torch.nn.ReLU()

        self.set_task(0)

    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        
    def set_task(self, id):
        self.current_task = id

    def forward(self, x, sample=False):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h))
        h=self.relu(self.conv6(h))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h)))
        y = self.last[self.current_task](h)
        
        return y
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10):
        super(AlexNet, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = nn.Linear(256 * s * s, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(4096, num_classes)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))
            
        self.relu = nn.ReLU()
        self.set_task(0)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        
    def set_task(self, id):
        self.current_task = id

    def _backbone(self, x, sample=False):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))
        
        x = self.relu(self.conv4(x))
        
        x = self.relu(self.conv5(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return x
        
    def forward(self, x, sample=False):
        x = self._backbone(x, sample)
        x = self.last[self.current_task](x)
        return x
    
class MultiHeadAlexNet(AlexNet):
    def __init__(self, inputsize, num_heads=1, num_classes=10):
        super(MultiHeadAlexNet, self).__init__(inputsize, num_heads, num_classes)
    
    def forward(self, x, sample=False):
        x = self._backbone(x, sample)
        outputs = torch.cat([head(x).unsqueeze(1) for head in self.last], axis=1)
        return outputs

import torch
import torch.nn as nn

class AlexNetV2(nn.Module):
    def __init__(self, inputsize, num_heads=1, num_classes=10):
        super(AlexNetV2, self).__init__()

        ncha, size, _ = inputsize
        
        self.conv1 = nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        s = size // 8  # considering three max pool layers each reducing size by half
        
        self.fc1 = nn.Linear(256 * s * s, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.ModuleList()
        for _ in range(num_heads):
            self.last.append(nn.Linear(4096, num_classes))
            
        self.relu = nn.ReLU()
        self.set_task(0)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        
    def set_task(self, id):
        self.current_task = id
        
    def forward(self, x, sample=False):
        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.conv3_bn(self.conv3(x)))
        
        x = self.relu(self.conv4_bn(self.conv4(x)))
        
        x = self.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))

        x = self.last[self.current_task](x)
        
        return x
    

class AlexNet64(nn.Module):
    def __init__(self, inputsize, num_classes=20):
        super(AlexNet64, self).__init__()

        ncha, size, _ = inputsize
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(ncha, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size after conv and pool layers
        s = size // 8  # Size is reduced by half at each of the three pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * s * s, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(0.5)
        
        self.last = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        
        x = torch.relu(self.conv4(x))
        
        x = torch.relu(self.conv5(x))
        x = self.pool3(x)
        
        # Flatten the output of the convolutional layers for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with dropout
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        
        # Output layer
        x = self.last(x)
        
        return x
