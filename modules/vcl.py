from torch import nn
import ast
from modules import VCLBayesianLinear
from .utils import get_activation_function


class VCL(nn.Module):

    def __init__(self, input_size, output_size, layers, activation_fn='relu', n_heads=10, mle_model=None):
        super(VCL, self).__init__()
        self.layers = nn.ModuleList()
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(VCLBayesianLinear(input_size, layers[0]))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(VCLBayesianLinear(layers[i], layers[i+1]))
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(VCLBayesianLinear(layers[-1], output_size))

        # If a MLE model is passed, the copy the weights
        if mle_model:
            self._copy_weights(mle_model)
        
        self.set_task(0)

    def _copy_weights(self, mle_model):
        vcl_linears = []
        mle_linears = []
        for vcl in self.modules():
            if isinstance(vcl, VCLBayesianLinear):
                vcl_linears.append(vcl)
        for mle in mle_model.modules():
            if isinstance(mle, nn.Linear):
                mle_linears.append(mle)
            
        for vcl, mle in zip(vcl_linears, mle_linears):
            vcl.posterior_weights.mean.data.copy_(mle.weight.data.view(-1))
            vcl.posterior_biases.mean.data.copy_(mle.bias.data.view(-1))
            vcl.prev_post_weights.mean.data.copy_(mle.weight.data.view(-1))
            vcl.prev_post_biases.mean.data.copy_(mle.bias.data.view(-1))

    def set_task(self, id):
        self.current_task = id

    def kl_div(self):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, VCLBayesianLinear):
                kl_div += layer.posterior_kl_div()
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.layers:
            if isinstance(layer, VCLBayesianLinear):
                layer.update_posterior()

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, VCLBayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        x = self.heads[self.current_task](x, sample)
        return x

