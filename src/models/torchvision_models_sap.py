import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import BaseModel

import torch.distributed as dist

from typing import Union, List
from .tv_resnet_sap import resnet18, resnet34, resnet50, resnet101, resnet152
from .tv_vit_sap import vit_b_32, vit_b_16, vit_l_32, vit_l_16, vit_h_14


class TorchvisionModelsSAP(BaseModel):
    
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
            model_func = resnet18
        elif model_type.startswith('resnet34'):
            model_func = resnet34
        elif model_type.startswith('resnet50'):
            model_func = resnet50
        elif model_type.startswith('resnet101'):
            model_func = resnet101
        elif model_type.startswith('vit_b_16'):
            model_func = vit_b_16
        elif model_type.startswith('vit_b_32'):
            model_func = vit_b_32
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
    
    
    def get_activations(self, x, prev_recur_proj_mat=None):
        return self.net.get_activations(x, prev_recur_proj_mat)
    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        return self.net.project_weights(projection_mat_dict, proj_classifier)
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Returns a module that maps input -> pre-logit features
        (exactly the tensor consumed by the final Linear head).
        """
        if self.model_type.startswith('resnet'):
            # All children except the final fc; add Flatten to match fc input
            backbone = nn.Sequential(*list(self.net.children())[:-1], nn.Flatten(1))
            return backbone

        elif self.model_type.startswith('vit'):
            # Mirror torchvision ViT forward up to heads (pre-classifier)
            class _ViTFeatureExtractor(nn.Module):
                def __init__(self, vit):
                    super().__init__()
                    self.vit = vit  
                def forward(self, x):
                    x = self.vit._process_input(x)                
                    n = x.shape[0]
                    cls = self.vit.class_token.expand(n, -1, -1)  
                    x = torch.cat((cls, x), dim=1)                
                    x = self.vit.encoder(x)                      
                    if self.vit._classifier == "token":
                        x = x[:, 0]                               
                    elif self.vit._classifier == "gap":
                        x = x.mean(dim=1)                         
                    return x

            return _ViTFeatureExtractor(self.net)

        else:
            raise ValueError(f"Unsupported model_type for feature extractor: {self.model_type}")

    def get_classifier_head(self) -> nn.Module:
        """
        Returns the final Linear classification head module.
        """
        if self.model_type.startswith('resnet'):
            return self.net.fc
        elif self.model_type.startswith('vit'):
            return self.net.heads.head
        else:
            raise ValueError(f"Unsupported model_type for classifier head: {self.model_type}")

    
    
    def get_identifier(self):
        return 'Torchvision Model with SAP ' + self.model_type
    
    
    
    def _replace_bn_with_identity(self, module):
        """Recursively replace all BatchNorm layers with Identity."""
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                setattr(module, name, nn.Identity())
            else:
                self._replace_bn_with_identity(child)