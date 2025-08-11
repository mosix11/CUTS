import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import BaseClassificationModel

from typing import Union, List

class TorchvisionModels(BaseClassificationModel):
    
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
        
        net = None
        
        # TODO check for img_size and grayscalse and modify models
        if model_type == 'resnet18':
            net = torchvision.models.resnet18(weights=pt_weights, num_classes=num_classes)
                
        if model_type == 'resnet18_nonorm':
            net = torchvision.models.resnet18(norm_layer=nn.Identity, weights=pt_weights, num_classes=num_classes)
            net.conv1 = nn.Conv2d(1 if grayscale else 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            net.maxpool = nn.Identity()

        elif model_type == 'resnet50':
            # net = torchvision.models.resnet50(weights=pt_weights, num_classes=num_classes)
            net = torchvision.models.resnet50(weights=pt_weights)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif model_type == 'resnet50_nonorm':
            if pt_weights:
                net = torchvision.models.resnet50(weights=pt_weights)
                self._replace_bn_with_identity(net)
            else:
                net = torchvision.models.resnet50(norm_layer=nn.Identity)
                
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            
        elif model_type == 'vit_b_16':
            net = torchvision.models.vit_b_16(weights=pt_weights)
            net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
                
        elif model_type == 'vit_b_32':
            net = torchvision.models.vit_b_32(weights=pt_weights, num_classes=num_classes)
        else:
            raise ValueError(f"The model type {model_type} is not valid.")    
        
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