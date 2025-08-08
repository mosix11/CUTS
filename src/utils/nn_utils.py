import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
from typing import Any
import math














def calculate_ouput_dim(net, input_dim=None):

    batch_size = 8
    for child in net.children():
        if isinstance(child, nn.Sequential):
            child = child[0]
        for layer in child.modules():
            if isinstance(layer, nn.Sequential):
                layer = layer[0]

            if isinstance(layer, nn.Linear):
                (W_1, in_size) = list(net.parameters())[0].shape
                dummy_input = torch.randn((batch_size, in_size))
                output = net(dummy_input)
            elif isinstance(layer, nn.Conv2d):
                (K, in_C, K_h, K_w) = list(net.parameters())[0].shape
                if input_dim == None:
                    dummy_input = torch.randn(batch_size, in_C, 1024, 1024)
                    print(
                        "Since the input dimension was not specified the default input size was set to be (1024, 1024)"
                    )
                else:
                    if input_dim[0] != in_C:
                        error_str = "input channels {} doesnt match the input channel size of network {}".format(
                            input_dim[0], in_C
                        )
                        raise Exception(error_str)
                    dummy_input = torch.randn(
                        batch_size, in_C, input_dim[1], input_dim[2]
                    )

                output = net(dummy_input)
            break
        break

    return output.shape






def compute_grad_norm_stats(model):
    # Collect all gradient norms
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:  # Ensure the parameter has a gradient
            grad_norms.append(
                param.grad.norm(2).item()
            )  # Compute the L2 norm of the gradient

    if grad_norms:
        max_grad_norm = max(grad_norms)  # Maximum gradient norm
        avg_grad_norm = sum(grad_norms) / len(grad_norms)  # Average gradient norm
    else:
        max_grad_norm = 0.0
        avg_grad_norm = 0.0

    return max_grad_norm, avg_grad_norm