from torch import nn

def get_activation_function(name):
    """Return the specified activation function from the torch.nn module."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f'Unknown activation function: {name}')