from torch import nn
import ast
from modules import TDBayesianLinear, NStepKLVCL
from .utils import get_activation_function

class TemporalDifferenceVCL(NStepKLVCL):
    def __init__(self, input_size, output_size, n_step, lambd, layers, activation_fn='relu', n_heads=10, lambda_logvar=-5.0):
        super(TemporalDifferenceVCL, self).__init__(input_size, output_size, n_step, layers, activation_fn, n_heads, lambda_logvar)
        self.layers = nn.ModuleList()
        self.n_step = n_step
        self.lambd = lambd
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(TDBayesianLinear(input_size, layers[0], n_step, lambd, lambda_logvar))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(TDBayesianLinear(layers[i], layers[i+1], n_step, lambd, lambda_logvar))
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(TDBayesianLinear(layers[-1], output_size, n_step, lambd, lambda_logvar))
        
        self.set_task(0)

