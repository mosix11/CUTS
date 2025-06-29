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
        metrics:dict=None,
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Layer 0: nn.Conv2d
        # Layer 1: nn.BatchNorm2d
        # Layer 2: nn.ReLU
        # Layer 3: nn.Conv2d
        # Layer 4: nn.BatchNorm2d
        # Layer 5: nn.ReLU
        # Layer 6: nn.MaxPool2d  (This takes up an index)
        # Layer 7: nn.Conv2d
        # Layer 8: nn.BatchNorm2d
        # Layer 9: nn.ReLU
        # Layer 10: nn.MaxPool2d (This takes up an index)
        # Layer 11: nn.Conv2d
        # Layer 12: nn.BatchNorm2d
        # Layer 13: nn.ReLU
        # Layer 14: nn.MaxPool2d (This takes up an index)
        # Layer 15: nn.MaxPool2d (This takes up an index)
        # Layer 16: Flatten()    (This takes up an index)
        # Layer 17: nn.Linear    (This is your last layer)
            
        
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
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance
    
    
    def training_step(self, x, y, use_amp=False, return_preds=False):
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
    
    def forward(self, x):
        return self.net(x)
    
    
    def load_backbone_weights(self, state_dict):
        """
        Loads weights into the backbone layers (all layers except the last classification head)
        from a given state_dict, typically from another trained model of the same class.
        
        Args:
            state_dict (dict): The state dictionary from another CNN5 model.
        """
        
        # Filter out the last layer's weights from the loaded state_dict
        pretrained_backbone_state_dict = {
            k: v for k, v in state_dict.items() 
            if not (k.startswith("net.17.weight") or k.startswith("net.17.bias"))
        }
        
        self.load_state_dict(pretrained_backbone_state_dict, strict=False)
        print("Backbone weights loaded successfully.")
    
        
    def get_backbone_weights(self):
        # Filter out the last layer's weights from the loaded state_dict
        backbone_state_dict = {
            k: v for k, v in self.state_dict().items() 
            if not (k.startswith("net.17.weight") or k.startswith("net.17.bias"))
        }
        
        return backbone_state_dict
    
    
    def freeze_classification_head(self):
        """
        Freezes the weights of the last linear layer (classification head).
        """
        # The last layer is at index -1 in the nn.Sequential module
        last_layer = self.net[-1] 
        if isinstance(last_layer, nn.Linear):
            for param in last_layer.parameters():
                param.requires_grad = False
            print("Last layer (classification head) weights frozen.")
        else:
            print("The last layer is not an nn.Linear layer. No weights frozen.")

    def unfreeze_classification_head(self):
        """
        Unfreezes the weights of the last linear layer (classification head).
        """
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            for param in last_layer.parameters():
                param.requires_grad = True
            print("Last layer (classification head) weights unfrozen.")
        else:
            print("The last layer is not an nn.Linear layer.")
    
    
    def freeze_backbone(self):
        """
        Freezes the weights of all layers except the last linear classification head.
        """
        for name, param in self.named_parameters():
            # Correctly identify the last layer's parameters
            if not (name.startswith("net.17.weight") or name.startswith("net.17.bias")):
                param.requires_grad = False
        print("Backbone layers (all except the last classification head) frozen.")

    def unfreeze_backbone(self):
        """
        Unfreezes the weights of all layers except the last linear classification head.
        """
        for name, param in self.named_parameters():
            if not (name.startswith("net.17.weight") or name.startswith("net.17.bias")):
                param.requires_grad = True
        print("Backbone layers (all except the last classification head) unfrozen.")
    
    
    
    def get_identifier(self):
        return f"cnn5|k{self.num_channels}"
    
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    
    
class CNN5_NoNorm(nn.Module):
    
    def __init__(
        self,
        num_channels:int = 64,
        num_classes: int = 10,
        gray_scale: bool = False,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss,
        metrics:dict=None,
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Layer 0: nn.Conv2d
        # Layer 1: nn.BatchNorm2d
        # Layer 2: nn.ReLU
        # Layer 3: nn.Conv2d
        # Layer 4: nn.BatchNorm2d
        # Layer 5: nn.ReLU
        # Layer 6: nn.MaxPool2d  (This takes up an index)
        # Layer 7: nn.Conv2d
        # Layer 8: nn.BatchNorm2d
        # Layer 9: nn.ReLU
        # Layer 10: nn.MaxPool2d (This takes up an index)
        # Layer 11: nn.Conv2d
        # Layer 12: nn.BatchNorm2d
        # Layer 13: nn.ReLU
        # Layer 14: nn.MaxPool2d (This takes up an index)
        # Layer 15: nn.MaxPool2d (This takes up an index)
        # Layer 16: Flatten()    (This takes up an index)
        # Layer 17: nn.Linear    (This is your last layer)
            
        
        self.net = nn.Sequential(
            # Layer 0
            nn.Conv2d(1 if gray_scale else 3, num_channels, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
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
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance
    
    
    def training_step(self, x, y, use_amp=False, return_preds=False):
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
    
    def forward(self, x):
        return self.net(x)
    
    
    def load_backbone_weights(self, state_dict):
        """
        Loads weights into the backbone layers (all layers except the last classification head)
        from a given state_dict, typically from another trained model of the same class.
        
        Args:
            state_dict (dict): The state dictionary from another CNN5 model.
        """
        
        # Filter out the last layer's weights from the loaded state_dict
        pretrained_backbone_state_dict = {
            k: v for k, v in state_dict.items() 
            if not (k.startswith("net.17.weight") or k.startswith("net.17.bias"))
        }
        
        self.load_state_dict(pretrained_backbone_state_dict, strict=False)
        print("Backbone weights loaded successfully.")
    
        
    def get_backbone_weights(self):
        # Filter out the last layer's weights from the loaded state_dict
        backbone_state_dict = {
            k: v for k, v in self.state_dict().items() 
            if not (k.startswith("net.17.weight") or k.startswith("net.17.bias"))
        }
        
        return backbone_state_dict
    
    
    def freeze_classification_head(self):
        """
        Freezes the weights of the last linear layer (classification head).
        """
        # The last layer is at index -1 in the nn.Sequential module
        last_layer = self.net[-1] 
        if isinstance(last_layer, nn.Linear):
            for param in last_layer.parameters():
                param.requires_grad = False
            print("Last layer (classification head) weights frozen.")
        else:
            print("The last layer is not an nn.Linear layer. No weights frozen.")

    def unfreeze_classification_head(self):
        """
        Unfreezes the weights of the last linear layer (classification head).
        """
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            for param in last_layer.parameters():
                param.requires_grad = True
            print("Last layer (classification head) weights unfrozen.")
        else:
            print("The last layer is not an nn.Linear layer.")
    
    
    def freeze_backbone(self):
        """
        Freezes the weights of all layers except the last linear classification head.
        """
        for name, param in self.named_parameters():
            # Correctly identify the last layer's parameters
            if not (name.startswith("net.17.weight") or name.startswith("net.17.bias")):
                param.requires_grad = False
        print("Backbone layers (all except the last classification head) frozen.")

    def unfreeze_backbone(self):
        """
        Unfreezes the weights of all layers except the last linear classification head.
        """
        for name, param in self.named_parameters():
            if not (name.startswith("net.17.weight") or name.startswith("net.17.bias")):
                param.requires_grad = True
        print("Backbone layers (all except the last classification head) unfrozen.")
    
    
    
    def get_identifier(self):
        return f"cnn5|k{self.num_channels}"
    
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    
    
