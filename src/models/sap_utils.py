# Code from https://github.com/sangamesh-kodge/SAP/blob/master/models/model_utils.py

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import torch 
import copy

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/float(stride[0])+1)), int(np.floor((Lin[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/float(stride[1])+1))


def reshape_conv_input_activation(x, conv_layer=None, kernel_size=3, stride=1, padding=0, dilation=1):
    ### FAST CODE (Avoid for loops)
    if conv_layer:
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding =  conv_layer.padding 
        dilation = conv_layer.dilation
    x_unfold = torch.nn.functional.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    mat = x_unfold.permute(0,2,1).contiguous().view(-1,x_unfold.shape[1])
    return mat

def forward_cache_activations(x, layer, key, prev_recur_proj_mat=None, act={"pre":OrderedDict(), "post":OrderedDict()} ):       
    if isinstance(layer, nn.Conv2d):
        if prev_recur_proj_mat is not None:
            act["pre"][key]=torch.matmul(reshape_conv_input_activation(deepcopy(x.clone().detach()), layer), prev_recur_proj_mat["pre"][key]).cpu().numpy()
            # Easier to project weights and then convolve.
            weight =torch.mm(layer.weight.data.flatten(1), prev_recur_proj_mat["pre"][key].transpose(0,1)).view_as(layer.weight.data)
            bias = None if layer.bias is None else layer.bias.data
            stride = layer.stride
            padding =  layer.padding 
            dilation = layer.dilation
            x = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=1)
            act["post"][key]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))
            x = torch.matmul( x.permute(0,2,3,1).reshape(-1, x.shape[1]).contiguous(), prev_recur_proj_mat["post"][key] ).reshape(x.shape[0],x.shape[2], x.shape[3], x.shape[1]).permute(0,3,1,2).contiguous()
        else:
            act["pre"][key]=reshape_conv_input_activation(deepcopy(x.clone().detach()), layer).cpu().numpy()
            x = layer(x)
            act["post"][key]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))                            
    elif isinstance(layer, nn.Linear):
        if prev_recur_proj_mat is not None:
            act["pre"][key]=torch.matmul(deepcopy(x.clone().detach()), prev_recur_proj_mat["pre"][key]).cpu().numpy()
            x = torch.matmul( x, prev_recur_proj_mat["pre"][key] )     
            x = layer(x)
            act["post"][key]= deepcopy(x.clone().detach().cpu().numpy()) 
            x = torch.matmul( x, prev_recur_proj_mat["post"][key] )            
        else:
            act["pre"][key]=deepcopy(x.clone().detach().cpu().numpy())
            x = layer(x)
            act["post"][key]= deepcopy(x.clone().detach().cpu().numpy())  
    else:
        x = layer(x)
    return act, x


def auto_get_activations(x, layers, block_key, prev_recur_proj_mat, act):
    if isinstance(layers, nn.Sequential):
        layer_ind = 0
        for layer in layers:
            layer_key = f"{block_key}.layer{layer_ind}"
            act, x = get_activations_layer(x, layer, layer_key, prev_recur_proj_mat, act)
            layer_ind+=1 
        return act, x
    else:
        return get_activations_layer(x, layers, block_key, prev_recur_proj_mat, act)

def get_activations_layer(x, layer, layer_key, prev_recur_proj_mat, act):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or hasattr(layer, "get_activations") ) :
        x = layer(x)
        return act, x
    if hasattr(layer, "get_activations"):
        return layer.get_activations(x, layer_key, prev_recur_proj_mat, act)  
    else:
        return forward_cache_activations(x, layer, layer_key, prev_recur_proj_mat, act) 

def auto_project_weights(layers, block_key, projection_mat_dict, proj_classifier = True):
    if isinstance(layers, nn.Sequential):
        layer_ind = 0
        for layer_number, layer in enumerate(layers):
            layer_key = f"{block_key}.layer{layer_ind}"
            if layer_number == len(layers)-1 :
                project_weights_layer(layer, layer_key, projection_mat_dict, proj_classifier)
            else:
                project_weights_layer(layer, layer_key, projection_mat_dict, True)
            layer_ind+=1 
    else:
        project_weights_layer(layers, block_key, projection_mat_dict, proj_classifier)
    return 

def project_weights_layer(layer, layer_key, projection_mat_dict, post_projection = True):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or hasattr(layer, "project_weights") ) :
        return
    if hasattr(layer, "project_weights"):
        layer.project_weights(layer_key, projection_mat_dict)  
    else:
        if post_projection:
            layer.weight.data = torch.mm(projection_mat_dict["post"][f"{layer_key}"].transpose(0,1) ,torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"{layer_key}"].transpose(0,1))).view_as(layer.weight.data)
            if layer.bias is not None:
                layer.bias.data = torch.mm( layer.bias.data.unsqueeze(0), projection_mat_dict["post"][f"{layer_key}"]).squeeze(0)
        else:
            layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"{layer_key}"].transpose(0,1)).view_as(layer.weight.data)
    

    



