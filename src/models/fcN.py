import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class FCN(nn.Module):
    
    def __init__(
        self,
        input_dim=784,
        h_dims:list[int] = None,
        output_dim=10,
        weight_init=None,
        loss_fn=None,
        metrics:dict=None,
    ):
        super().__init__()
        
        
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
        
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        
            
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance
            
    

    def training_step(self, x, y, use_amp=False):        
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            if isinstance(self.loss_fn, torch.nn.MSELoss):
                y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                loss = self.loss_fn(preds, y_onehot)
            else:
                loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        return loss
        
    def validation_step(self, x, y, use_amp=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                if isinstance(self.loss_fn, torch.nn.MSELoss):
                    y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                    loss = self.loss_fn(preds, y_onehot)
                else:
                    loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        return loss


    def compute_metrics(self):
        results = {}
        if self.metrics: 
            for name, metric in self.metrics.items():
                results[name] = metric.compute()
        return results
    
    def reset_metrics(self):
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.h_first(x))
        for layer in self.middle_layers: 
            x = F.relu(layer(x))
        x = self.out(x)
        return x
    
    def get_identifier(self):
        identifier = f"fc{len(self.h_dims)}|h{str(tuple(self.h_dims))}|p{self._count_trainable_parameters()}"
        return identifier
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)