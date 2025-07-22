import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import BaseClassificationModel

class TorchvisionModels(BaseClassificationModel):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        num_classes:int = None,
        weight_init = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        net = None
        
        if model_type == 'resnet18':
            net = torchvision.models.resnet18(weights=pt_weights)
            if num_classes and net.fc.out_features != num_classes:
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, num_classes)
                
        elif model_type == 'resnet50':
            net = torchvision.models.resnet50(weights=pt_weights)
            if num_classes and net.fc.out_features != num_classes:
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, num_classes)
                
        elif model_type == 'vit_b_16':
            net = torchvision.models.vit_b_16(weights=pt_weights)
            if num_classes and net.heads.head.out_features != num_classes:
                num_ftrs = net.heads.head.in_features
                net.heads.head = nn.Linear(num_ftrs, num_classes)
                
        elif model_type == 'vit_b_32':
            net = torchvision.models.vit_b_32(weights=pt_weights)
            if num_classes and net.heads.head.out_features != num_classes:
                num_ftrs = net.heads.head.in_features
                net.heads.head = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"The model type {model_type} is not valid.")    
        
        self.net = net
        
        
        if weight_init:
            self.apply(weight_init)
            
    
    def forward(self, x):
        return self.net(x)
    
    
    def get_identifier(self):
        return 'Torchvision Model'