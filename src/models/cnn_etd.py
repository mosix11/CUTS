import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .etd import ExampleTiedDropout

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))



class CNN5_ETD(nn.Module):
    
    def __init__(
        self,
        num_channels:int = 64,
        num_classes: int = 10,
        gray_scale: bool = False,
        dropout: dict = None,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss,
        metrics:dict=None,
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1 if gray_scale else 3, num_channels, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*2),
            nn.ReLU(),
        )
        
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*4),
            nn.ReLU(),
        )
        
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*8),
            nn.ReLU(),
        )
        
        self.flatten = Flatten()
        self.fc = nn.Linear(num_channels*8, num_classes, bias=True)
        
        if dropout:
            self.dropout1 = ExampleTiedDropout(**dropout, num_channels=num_channels)
            self.dropout2 = ExampleTiedDropout(**dropout, num_channels=num_channels*2)
            self.dropout3 = ExampleTiedDropout(**dropout, num_channels=num_channels*4)
            self.dropout4 = ExampleTiedDropout(**dropout, num_channels=num_channels*8)
        else:
            self.dropout1 = self.dropout2 = self.dropout3 = self.dropout4 = None


        if weight_init:
            self.apply(weight_init)
            
            
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance
    
    
    def training_step(self, x, y, indices, use_amp=False, return_preds=False):
        with autocast('cuda', enabled=use_amp):
            preds = self(x, indices)
            loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        if return_preds:
            return loss, preds
        else:
            return loss
        
    def validation_step(self, x, y, use_amp=False, return_preds=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        if return_preds:
            return loss, preds
        else:
            return loss


    def compute_metrics(self):
        results = {}
        if self.metrics: 
            for name, metric in self.metrics.items():
                results[name] = metric.compute().cpu().item()
        return results
    
    def reset_metrics(self):
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, X, indices=None):
        X = self.layer1(X)
        X = self.dropout1(X, indices)
        X = self.layer2(X)
        X = self.dropout2(X, indices)
        X = F.max_pool2d(X, kernel_size=2)
        X = self.layer3(X)
        X = self.dropout3(X, indices)
        X = F.max_pool2d(X, kernel_size=2)
        X = self.layer4(X)
        X = self.dropout4(X, indices)
        X = F.max_pool2d(X, kernel_size=2)
        X = F.max_pool2d(X, kernel_size=4)
        X = self.flatten(X)
        X = self.fc(X)
        return X
    
    
    def get_identifier(self):
        return f"cnn5|k{self.num_channels}"
    
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    