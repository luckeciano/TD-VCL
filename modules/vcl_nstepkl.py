from torch import nn
import ast
from .nstepkl_bayesian_linear import NStepKLVCLBayesianLinear
from .utils import get_activation_function

class NStepKLVCL(nn.Module):
    def __init__(self, input_size, output_size, n_step, layers, activation_fn='relu', n_heads=10):
        super(NStepKLVCL, self).__init__()
        self.layers = nn.ModuleList()
        self.n_step = n_step
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(NStepKLVCLBayesianLinear(input_size, layers[0], n_step))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(NStepKLVCLBayesianLinear(layers[i], layers[i+1], n_step))
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(NStepKLVCLBayesianLinear(layers[-1], output_size, n_step))
        
        self.set_task(0)

    def set_task(self, id):
        self.current_task = id

    def kl_div(self, curr_timestep):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, NStepKLVCLBayesianLinear):
                kl_div += layer.posterior_kl_div(curr_timestep)
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.layers:
            if isinstance(layer, NStepKLVCLBayesianLinear):
                layer.update_posterior()

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, NStepKLVCLBayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        x = self.heads[self.current_task](x, sample)
        return x

