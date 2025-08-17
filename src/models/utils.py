import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
from typing import Any
import math


def init_constant(module, const):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)


def init_uniform(module, l=0, u=1):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)


def init_normal(module, mean=0, std=0.01):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)


def init_xavier_uniform(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


def init_xavier_normal(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_normal_(module._parameters[param])


def lazy_layer_initialization(model, dummy_input, init_method=None):
    model(*dummy_input)
    if init_method is not None:
        model.apply(init_method)
        
        