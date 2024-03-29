from torch import nn
import ast
from .utils import get_activation_function

class MultiHeadMLP(nn.Module):

    def __init__(self, input_size, output_size, layers, activation_fn='relu', n_heads=10):
        super(MultiHeadMLP, self).__init__()
        self.layers = nn.ModuleList()
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(nn.Linear(input_size, layers[0]))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(nn.Linear(layers[-1], output_size))
        
        self.set_task(0)

    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        
    def set_task(self, id):
        self.current_task = id

    def forward(self, x, sample=False):
        for layer in self.layers:
            x = layer(x)
        x = self.heads[self.current_task](x)
        return x

