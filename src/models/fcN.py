import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseClassificationModel


class FCN(BaseClassificationModel):
    
    def __init__(
        self,
        input_dim=784,
        h_dims:list[int] = None,
        output_dim=10,
        weight_init=None,
        loss_fn=None,
        metrics:dict=None,
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        
        if len(h_dims) < 2:
            raise ValueError('This module is designed for networks deeper than 2 hidden layers.')
        
        self.input_dim = input_dim
        self.h_dims = h_dims
        self.output_dim = output_dim
        
        self.h_first = nn.Linear(input_dim, self.h_dims[0], bias=True)
        self.out = nn.Linear(self.h_dims[-1], output_dim, bias=True)
        
        self.middle_layers = nn.ModuleList()
        
        for i in range(len(h_dims) - 1):
            self.middle_layers.append(nn.Linear(h_dims[i], h_dims[i+1], bias=True))
            
        
        
        if weight_init:
            self.apply(weight_init)
        
        
    def forward(self, x: torch.Tensor):
        x = F.relu(self.h_first(x))
        for layer in self.middle_layers: 
            x = F.relu(layer(x))
        x = self.out(x)
        return x
    
    def get_identifier(self):
        identifier = f"fc{len(self.h_dims)}|h{str(tuple(self.h_dims))}|p{self._count_trainable_parameters()}"
        return identifier
    
    