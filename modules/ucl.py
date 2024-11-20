from torch import nn
import ast
from modules import UCLBayesianLinear
from .utils import get_activation_function


class UCL(nn.Module):
    def __init__(self, input_size, output_size, layers, activation_fn='relu', n_heads=10, lambd_logvar=-5.0, mle_model=None, ratio=0.5, alpha=0.01, beta=0.03, gamma=0.0):
        super(UCL, self).__init__()
        self.layers = nn.ModuleList()
        layers = ast.literal_eval(layers)
        self.bayesian_layers = []
        
        # Input layer
        layer = UCLBayesianLinear(input_size, layers[0], lambd_logvar, ratio, alpha, beta, gamma)
        self.bayesian_layers.append(layer)
        self.layers.append(layer)
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            layer = UCLBayesianLinear(layers[i], layers[i+1], lambd_logvar, ratio, alpha, beta, gamma, self.bayesian_layers[-1])
            self.bayesian_layers.append(layer)
            self.layers.append(layer)
        
        self.layers.append(get_activation_function(activation_fn))
        
        # Output Heads
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            self.heads.append(UCLBayesianLinear(layers[-1], output_size, lambd_logvar, ratio, alpha, beta, gamma, self.bayesian_layers[-1]))

        # If a MLE model is passed, the copy the weights
        if mle_model:
            self._copy_weights(mle_model)
        
        self.set_task(0)

    def _copy_weights(self, mle_model):
        ucl_linears = []
        mle_linears = []
        for ucl in self.modules():
            if isinstance(ucl, UCLBayesianLinear):
                ucl_linears.append(ucl)
        for mle in mle_model.modules():
            if isinstance(mle, nn.Linear):
                mle_linears.append(mle)
            
        for ucl, mle in zip(ucl_linears, mle_linears):
            ucl.posterior_weights.mean.data.copy_(mle.weight.data.view(-1))
            ucl.posterior_biases.mean.data.copy_(mle.bias.data.view(-1))
            ucl.prev_post_weights.mean.data.copy_(mle.weight.data.view(-1))
            ucl.prev_post_biases.mean.data.copy_(mle.bias.data.view(-1))

    def set_task(self, id):
        self.current_task = id

    def kl_div(self):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, UCLBayesianLinear):
                kl_div += layer.posterior_kl_div()
        return kl_div
    
    def new_task(self, task_id, single_head):
        self.set_task(0 if single_head else task_id)
        for layer in self.layers:
            if isinstance(layer, UCLBayesianLinear):
                layer.update_posterior()

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, UCLBayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        x = self.heads[self.current_task](x, sample)
        return x

