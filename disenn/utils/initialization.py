import math
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, calculate_gain

def init_parameters(module):
    """Initialize parameters of a given layer-activation block"""
    # see also https://github.com/pytorch/pytorch/issues/18182
    supported_modules = {nn.Conv2d,
                         nn.Conv3d,
                         nn.ConvTranspose2d,
                         nn.ConvTranspose3d,
                         nn.Linear}
    for m in module.modules():
        if type(m) in supported_modules:
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)

# TODO: Update with the current best practice
def init_parameters_v0(layer, activation: str = "relu"):
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