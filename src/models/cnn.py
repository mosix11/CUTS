import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class CNN5(nn.Module):
    
    def __init__(
        self,
        num_channels:int = 64,
        num_classes: int = 10,
        gray_scale: bool = False,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss,
        metric=None,
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            # Layer 0
            nn.Conv2d(1 if gray_scale else 3, num_channels, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(num_channels*8, num_classes, bias=True)
        )

        if weight_init:
            self.apply(weight_init)
            
            
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        if metric:
            self.metric = metric
    
    
    def training_step(self, x, y, use_amp=False):
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
        
    def validation_step(self, x, y, use_amp=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
    
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, x):
        return self.net(x)
    
    
    def get_identifier(self):
        return f"cnn5|k{self.num_channels}"