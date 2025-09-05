import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import BaseModel

import torch.distributed as dist

from typing import Union, List

class TorchvisionModels(BaseModel):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        num_classes:int = None,
        img_size:Union[tuple, list] = None,
        grayscale: bool = False,
        weight_init = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        
        
        model_func = None
        if model_type.startswith('resnet18'):
            model_func = torchvision.models.resnet18
        elif model_type.startswith('resnet34'):
            model_func = torchvision.models.resnet34
        elif model_type.startswith('resnet50'):
            model_func = torchvision.models.resnet50
        elif model_type.startswith('resnet101'):
            model_func = torchvision.models.resnet101
        elif model_type.startswith('vit_b_16'):
            model_func = torchvision.models.vit_b_16
        elif model_type.startswith('vit_b_32'):
            model_func = torchvision.models.vit_b_32
        else:
            raise ValueError(f"The model type {model_type} is not valid.")    

        if self.is_distributed():
            if self.is_node_leader():
                # Construct once to trigger the download into cache
                _model = model_func(weights=pt_weights)
                del _model
            dist.barrier()

        net = model_func(weights=pt_weights)
        
        if model_type.endswith('nonorm'):
            self._replace_bn_with_identity(net)
            
        if model_type.startswith('resnet'):
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            if img_size == [32, 32]:
                net.conv1 = nn.Conv2d(1 if grayscale else 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                net.maxpool = nn.Identity()
        elif model_type.startswith('vit'):
            net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
            
        
        self.net = net
        
        
        if weight_init:
            self.apply(weight_init)
            
    
    def forward(self, x):
        return self.net(x)
    
    
    def get_identifier(self):
        return 'Torchvision Model ' + self.model_type
    
    
    
    def _replace_bn_with_identity(self, module):
        """Recursively replace all BatchNorm layers with Identity."""
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                setattr(module, name, nn.Identity())
            else:
                self._replace_bn_with_identity(child)