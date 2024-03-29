from torch import nn
import torch.nn.functional as F


class MultiHeadMNISTCNN(nn.Module):
    def __init__(self, output_size, n_heads=10):
        super(MultiHeadMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(nn.Linear(64, output_size))
        
        self.set_task(0)
    
    def set_task(self, id):
        self.current_task = id

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.heads[self.current_task](x)
        return x