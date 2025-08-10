import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import torchmetrics
from abc import ABC, abstractmethod



class BaseClassificationModel(nn.Module, ABC): 
    """
    Abstract Base Class for classification models, providing common
    training, validation, prediction, and metric handling functionalities.
    """
    def __init__(self, loss_fn=None, metrics: dict = None):
        super().__init__()

        if loss_fn is None:
            raise RuntimeError('The loss function must be specified for training/validation.')
        self.loss_fn = loss_fn

        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                if not isinstance(metric_instance, torchmetrics.Metric):
                    raise TypeError(f"Metric '{name}' must be an instance of torchmetrics.Metric.")
                self.metrics[name] = metric_instance
                
                
        self._NORM_LAYERS = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
            nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LocalResponseNorm
        )
        
        self._NORM_HINTS = [
            "batchnorm", "batch_norm", "bn",
            "layernorm", "layer_norm", "ln",
            "groupnorm", "group_norm", "gn",
            "instancenorm", "instance_norm", "inorm", "in"
        ]

    @abstractmethod
    def forward(self, x):
        """
        Abstract method that must be implemented by all concrete subclasses.
        This defines the specific architecture of the model.
        """
        pass
    
    @abstractmethod
    def get_identifier(self):
        """
        Abstract method that must be implemented by all concrete subclasses.
        This defines the specific architecture of the model.
        """
        pass

    def training_step(self, x, y, use_amp=False, return_preds=False):
        """Performs a single training step."""
        with autocast('cuda', enabled=use_amp):
            preds = self(x) 
            loss = self.loss_fn(preds, y)

        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds.detach(), y.detach()) 

        if return_preds:
            return loss, preds
        else:
            return loss


    @torch.no_grad()
    def validation_step(self, x, y, use_amp=False, return_preds=False):
        """Performs a single validation step (no gradient computation)."""
        
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            loss = self.loss_fn(preds, y)

        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds.detach(), y.detach())

        if return_preds:
            return loss, preds
        else:
            return loss

    @torch.no_grad()
    def predict(self, x):
        """Performs inference (prediction) without gradient computation."""
        preds = self(x)
        return preds

    def compute_metrics(self):
        """Computes and returns the current metric results."""
        results = {}
        if self.metrics:
            for name, metric in self.metrics.items():
                results[name] = metric.compute().cpu().item()
        return results

    def reset_metrics(self):
        """Resets all tracked metrics."""
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()

    
    
    def _count_trainable_parameters(self):
        """Counts and returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    