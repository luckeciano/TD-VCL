from torch import nn
import ast
from modules import TDUCLBayesianLinear
from .utils import get_activation_function

class TemporalDifferenceUCL(nn.Module):
    def __init__(self, input_size, output_size, n_step, lambd, layers, lambda_logvar = -5, ratio = 0.5, alpha = 0.01, beta = 0.03, gamma = 0, activation_fn='relu', n_heads=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_step = n_step
        layers = ast.literal_eval(layers)
        self.bayesian_layers = []
        
        # Input layer
        layer = TDUCLBayesianLinear(input_size, layers[0], n_step, lambd, lambda_logvar, ratio, alpha, beta, gamma, previous_ucl_layer=None)
        self.layers.append(layer)
        self.bayesian_layers.append(layer)
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            layer = TDUCLBayesianLinear(layers[i], layers[i+1], n_step, lambd, lambda_logvar, ratio, alpha, beta, gamma, previous_ucl_layer=self.bayesian_layers[-1])
            self.layers.append(layer)
            self.bayesian_layers.append(layer)
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(TDUCLBayesianLinear(layers[-1], output_size, n_step, lambd, lambda_logvar, ratio, alpha, beta, gamma, previous_ucl_layer=self.bayesian_layers[-1]))
        
        self.set_task(0)

    def set_task(self, id):
        self.current_task = id

    def kl_div(self, curr_timestep):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, TDUCLBayesianLinear):
                kl_div += layer.posterior_kl_div(curr_timestep)
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.layers:
            if isinstance(layer, TDUCLBayesianLinear):
                layer.update_posterior()

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, TDUCLBayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        x = self.heads[self.current_task](x, sample)
        return x

