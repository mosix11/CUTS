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
    
    
    
    def freeze_last_layer(self):
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

    def unfreeze_last_layer(self):
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
    
    
    def get_identifier(self):
        return f"cnn5|k{self.num_channels}"
    
    
    
    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    
    
    
    
    
    
    
class CNN5_BH(nn.Module):
    """
    A Convolutional Neural Network (CNN) with a primary classification head
    and an additional binary classification head.

    Args:
        num_channels (int): Base number of channels in the convolutional layers.
                            Channels will scale up (e.g., num_channels*2, num_channels*4, etc.).
                            Defaults to 64.
        num_classes (int): Number of output classes for the primary classification task.
                           Defaults to 10.
        gray_scale (bool): If True, the input expects 1 channel (grayscale images).
                           If False, expects 3 channels (RGB images).
                           Defaults to False.
        weight_init (callable, optional): A function to apply for custom weight initialization.
                                          If provided, `self.apply(weight_init)` is called.
                                          Defaults to None.
        loss_fn (torch.nn.Module): The loss function to be used. This should ideally be
                                   the CompoundLoss class or similar that handles multiple outputs.
                                   Defaults to nn.CrossEntropyLoss (though CompoundLoss is expected).
        metrics (dict, optional): A dictionary of metric instances (e.g., Accuracy).
                                  Metrics will be updated for the primary classification task.
                                  Defaults to None.
    """
    def __init__(
        self,
        num_channels:int = 64,
        num_classes: int = 10,
        gray_scale: bool = False,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss, # This will typically be CompoundLoss
        metrics:dict=None,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        
        self.feature_extractor = nn.Sequential(
            # Layer 0: Initial convolutional block
            nn.Conv2d(1 if gray_scale else 3, num_channels, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            # Layer 1: Doubles channels, adds MaxPool
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2), # Halves spatial dimensions

            # Layer 2: Doubles channels again, adds MaxPool
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*4),
            nn.ReLU(),
            nn.MaxPool2d(2), # Halves spatial dimensions again

            # Layer 3: Doubles channels one last time, adds MaxPool
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*8),
            nn.ReLU(),
            nn.MaxPool2d(2), # Halves spatial dimensions again

            # Layer 4: Final MaxPool and Flatten
            # The MaxPool2d(4) is applied here to further reduce spatial dimensions
            # to 1x1 if the input image is typically 32x32 (e.g., CIFAR-10)
            # (32 -> 16 -> 8 -> 4 after three MaxPool2d(2) -> 1 after MaxPool2d(4))
            nn.MaxPool2d(4),
            Flatten(),       # Flattens the 3D feature maps into a 1D vector
                             # for the subsequent linear layers.
        )

        # Primary Classification Head: For the `num_classes` task (e.g., image classification)
        # It takes the flattened features from the feature_extractor as input.
        # The input size `num_channels*8` comes from the output channels of the last
        # convolutional layer multiplied by the final spatial dimensions (which are 1x1).
        self.ce_head = nn.Linear(num_channels*8, num_classes, bias=True)

        # Binary Classification Head: For the binary task (e.g., noisy vs. clean sample detection)
        # It also takes the same flattened features from the feature_extractor as input.
        # Outputs a single logit (before sigmoid) for binary classification.
        self.binary_head = nn.Linear(num_channels*8, 1, bias=True)

        # Apply custom weight initialization if provided
        if weight_init:
            self.apply(weight_init)

        # Ensure a loss function is specified
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn

        # Initialize metrics as a ModuleDict
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance

    def training_step(self, x: torch.Tensor, y: tuple[torch.Tensor, torch.Tensor],
                      use_amp: bool = False, return_preds: bool = False) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            x (torch.Tensor): Input data (e.g., images).
            y (tuple[torch.Tensor, torch.Tensor]): A tuple containing
                                                   (ce_target, binary_target).
                                                   ce_target: ground truth labels for CE loss.
                                                   binary_target: ground truth labels for binary loss.
            use_amp (bool): If True, uses Automatic Mixed Precision for training.
            return_preds (bool): If True, returns predictions along with the loss.

        Returns:
            torch.Tensor: The calculated loss, and optionally the predictions.
        """
        # Unpack the targets for the two tasks
        ce_target, binary_target = y

        with autocast('cuda', enabled=use_amp):
            # Forward pass: the model now returns two outputs (CE logits, Binary logits)
            ce_preds, binary_preds = self(x)
            # Calculate the compound loss using the two sets of predictions and targets
            loss = self.loss_fn(ce_preds, ce_target, binary_preds, binary_target)

        if self.metrics:
            # Update metrics only for the primary classification task, as defined
            for name, metric in self.metrics.items():
                if 'CE' in name:
                    metric.update(ce_preds, ce_target)
                elif 'BN' in name:
                    metric.update(binary_preds.squeeze(), binary_target)
        if return_preds:
            # Return both losses and both predictions
            return loss, (ce_preds, binary_preds.squeeze())
        else:
            return loss

    def validation_step(self, x: torch.Tensor, y: tuple[torch.Tensor, torch.Tensor],
                        use_amp: bool = False, return_preds: bool = False) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            x (torch.Tensor): Input data (e.g., images).
            y (tuple[torch.Tensor, torch.Tensor]): A tuple containing
                                                   (ce_target, binary_target).
                                                   ce_target: ground truth labels for CE loss.
                                                   binary_target: ground truth labels for binary loss.
            use_amp (bool): If True, uses Automatic Mixed Precision for validation.
            return_preds (bool): If True, returns predictions along with the loss.

        Returns:
            torch.Tensor: The calculated loss, and optionally the predictions.
        """
        # Unpack the targets for the two tasks
        ce_target, binary_target = y

        with torch.no_grad(): # Disable gradient calculations for validation
            with autocast('cuda', enabled=use_amp):
                # Forward pass: the model now returns two outputs
                ce_preds, binary_preds = self(x)
                # Calculate the compound loss
                loss = self.loss_fn(ce_preds, ce_target, binary_preds, binary_target)

        if self.metrics:
            # Update metrics only for the primary classification task
            for name, metric in self.metrics.items():
                if 'CE' in name:
                    metric.update(ce_preds, ce_target)
                elif 'BN' in name:
                    metric.update(binary_preds.squeeze(), binary_target)
        if return_preds:
            # Return both losses and both predictions
            return loss, (ce_preds, binary_preds.squeeze())
        else:
            return loss

    def compute_metrics(self) -> dict:
        """
        Computes the current values of all registered metrics.

        Returns:
            dict: A dictionary where keys are metric names and values are their computed results.
        """
        results = {}
        if self.metrics:
            for name, metric in self.metrics.items():
                results[name] = metric.compute().cpu().item()
        return results

    def reset_metrics(self):
        """
        Resets the state of all registered metrics.
        """
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass without gradient calculation to get predictions.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing
                                               (ce_predictions, binary_predictions).
                                               ce_predictions: logits for primary classification.
                                               binary_predictions: logits for binary classification.
        """
        with torch.no_grad():
            ce_preds, binary_preds = self(x)
        return ce_preds, binary_preds

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): The input tensor (e.g., image batch).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing
                                               (ce_output, binary_output).
                                               ce_output: raw logits for multi-class classification.
                                               binary_output: raw logits for binary classification.
        """
        # Pass input through the shared feature extractor
        features = self.feature_extractor(x)
        # Pass features through the primary classification head
        ce_output = self.ce_head(features)
        # Pass features through the binary classification head
        binary_output = self.binary_head(features)
        return ce_output, binary_output

    def freeze_classification_heads(self):
        """
        Freezes the weights (sets requires_grad to False) of both the
        primary classification head (ce_head) and the binary classification head (binary_head).
        This is useful for transfer learning scenarios where you want to fine-tune
        the feature extractor first.
        """
        for param in self.ce_head.parameters():
            param.requires_grad = False
        for param in self.binary_head.parameters():
            param.requires_grad = False
        print("Classification (CE) and Binary heads weights frozen.")

    def unfreeze_classification_heads(self):
        """
        Unfreezes the weights (sets requires_grad to True) of both the
        primary classification head (ce_head) and the binary classification head (binary_head).
        """
        for param in self.ce_head.parameters():
            param.requires_grad = True
        for param in self.binary_head.parameters():
            param.requires_grad = True
        print("Classification (CE) and Binary heads weights unfrozen.")

    def get_identifier(self) -> str:
        """
        Generates a string identifier for the model configuration.
        """
        return f"cnn5|k{self.num_channels}"
    
    
    def load_pretrained_weights_from_old_cnn5(self, old_state_dict: dict):
        """
        Loads weights from a state dictionary of an older CNN5 model (with only a classification head)
        into the new CNN5 model (with feature_extractor, ce_head, and binary_head).

        Args:
            model (CNN5): The current CNN5 model instance.
            old_state_dict (dict): The state dictionary loaded from the old CNN5 model.
        """

        new_state_dict = self.state_dict()
        num_feature_extractor_layers = len(self.feature_extractor)

        for key, value in old_state_dict.items():
            if key.startswith('net.'):
                # Extract the module index from the old key, e.g., 'net.0.weight' -> 0
                parts = key.split('.')
                module_idx = int(parts[1])

                if module_idx < num_feature_extractor_layers:
                    # These keys belong to the feature_extractor
                    # Map 'net.X.param' to 'feature_extractor.X.param'
                    new_key = f'feature_extractor.{module_idx}.' + '.'.join(parts[2:])
                    if new_key in new_state_dict:
                        new_state_dict[new_key].copy_(value)
                    else:
                        print(f"Warning: Key {new_key} not found in current model. Skipping.")
                else:
                    # The remaining layer in 'net' is the final linear layer (classification head)
                    # Map 'net.X.param' to 'ce_head.param'
                    new_key = 'ce_head.' + '.'.join(parts[2:]) # e.g., 'ce_head.weight', 'ce_head.bias'
                    if new_key in new_state_dict:
                        new_state_dict[new_key].copy_(value)
                    else:
                        print(f"Warning: Key {new_key} not found in current model. Skipping.")
            else:
                print(f"Warning: Unrecognized key in old state_dict: {key}. Skipping.")

        # Load the updated state_dict into the model.
        # strict=False allows for missing keys (e.g., binary_head weights are new)
        self.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded successfully into feature_extractor and ce_head.")
        print("Binary head weights remain uninitialized (or randomly initialized if applicable).")

    def _count_trainable_parameters(self) -> int:
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
