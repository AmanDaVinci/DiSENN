import torch
from torch import nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, calculate_gain

def init_parameters(layer, activation: str = None):
    """Initialize parameters of a given layer-activation block

    Parameters
    ----------
    layer: nn.Linear or torch.nn.modules.conv._ConvNd
        layer whose parameters is to be initialized
    
    activation: torch.nn.modules.activation
        one of the activations: ["relu", "tanh", "sigmoid", None]
    """
    assert activation in ["relu", "tanh", "sigmoid", None],\
    "activation invalid or not supported"

    weight = layer.weight
    if activation is None:
        return xavier_uniform_(weight)
    elif activation == "relu":
        return kaiming_uniform_(weight, nonlinearity="relu")
    elif activation in ["sigmoid", "tanh"]:
        return xavier_uniform_(weight, gain=calculate_gain(activation))