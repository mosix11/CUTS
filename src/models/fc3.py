import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class FC3(nn.Module):
    
    def __init__(
        self,
        input_dim=784,
        h_dims=(16, 16, 16),
        output_dim=10,
        weight_init=None,
        loss_fn=None,
        metric=None,
    ):
        super().__init__()
        
        if not len(h_dims) == 3:
            raise ValueError('h_dims must contain three values specifying the width of three hidden layers.')
        
        self.input_dim = input_dim
        self.h1_dim = h_dims[0]
        self.h2_dim = h_dims[1]
        self.h3_dim = h_dims[2]
        self.output_dim = output_dim
        self.h1 = nn.Linear(input_dim, self.h1_dim, bias=True)
        self.h2 = nn.Linear(self.h1_dim, self.h2_dim, bias=True)
        self.h3 = nn.Linear(self.h2_dim, self.h3_dim, bias=True)
        self.out = nn.Linear(self.h3_dim, output_dim, bias=True)
        
        
        if weight_init:
            self.apply(weight_init)
        
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        if metric:
            self.metric = metric
            
    
    def reuse_weights(self, old_state: dict):
        """
        Load weights from a smaller FC3 model state_dict into this wider model.
        Copies the first `old_hidden` neurons exactly, and leaves the rest of the weights as they are initialized.

        Args:
            old_model_or_state: state dict of an FC3 instance.
            init_std: standard deviation for normal init of new weights.
        """
        # TODO implement this function
            
    def reuse_weights_and_freeze(self, old_state: dict):
        """
        Load weights from a smaller FC3 model state_dict into this wider model,
        and **freeze** the loaded weights (prevent them from training) using
        gradient hooks.

        Args:
            old_state: state dict of a smaller FC3 instance.
        """

        self.reuse_weights(old_state)

        # TODO implement this function


    def remove_freeze_hooks(self):
        """Removes any gradient hooks previously attached by reuse_weights_and_freeze."""
        if hasattr(self, '_freeze_handles'):
            for handle in self._freeze_handles:
                handle.remove()
        self._freeze_handles = []


    def log_stats(self):
        """
        Calculates and returns a dictionary containing the mean and standard deviation
        of the reused and non-reused weights and biases.

        Returns:
            dict: A dictionary where keys are parameter names (e.g., 'h1.weight_reused_mean')
                  and values are the corresponding statistics.
        """
    
        # TODO implement this function


    def training_step(self, x, y, use_amp=False):        
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            if isinstance(self.loss_fn, torch.nn.MSELoss):
                y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                loss = self.loss_fn(preds, y_onehot)
            else:
                loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
        
    def validation_step(self, x, y, use_amp=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                if isinstance(self.loss_fn, torch.nn.MSELoss):
                    y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                    loss = self.loss_fn(preds, y_onehot)
                else:
                    loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
    
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = self.out(x)
        return x
    
    def get_identifier(self):
        return f"fc3|h({self.h1_dim}, {self.h2_dim}, {self.h3_dim})|p{self._count_trainable_parameters()}"
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)