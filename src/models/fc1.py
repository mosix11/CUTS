import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class FC1(nn.Module):
    
    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        output_dim=10,
        weight_init=None,
        loss_fn=None,
        metrics:dict=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.h1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, output_dim, bias=True)
        
        
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
            if isinstance(self.loss_fn, torch.nn.MSELoss):
                y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                loss = self.loss_fn(preds, y_onehot)
            else:
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
                if isinstance(self.loss_fn, torch.nn.MSELoss):
                    y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                    loss = self.loss_fn(preds, y_onehot)
                else:
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
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.h1(x))
        x = self.out(x)
        return x
    
    def get_identifier(self):
        return f"fc1|h{self.hidden_dim}|p{self._count_trainable_parameters()}"
    
    
    # def get_backbone_weights(self):
    #     # Filter out the last layer's weights from the loaded state_dict
    #     backbone_state_dict = {
    #         k: v for k, v in self.state_dict().items() 
    #         if not (k.startswith("net.17.weight") or k.startswith("net.17.bias"))
    #     }
        
    #     return backbone_state_dict
    
    def freeze_classification_head(self):
        """
        Freezes the weights of the last linear layer (classification head).
        """
        # The last layer is at index -1 in the nn.Sequential module
        last_layer = self.out
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
        last_layer = self.out
        if isinstance(last_layer, nn.Linear):
            for param in last_layer.parameters():
                param.requires_grad = True
            print("Last layer (classification head) weights unfrozen.")
        else:
            print("The last layer is not an nn.Linear layer.")
    
    def reuse_weights(self, old_state: dict):
        """
        Load weights from a smaller FC1 model state_dict into this wider model.
        Copies the first `old_hidden` neurons exactly, and leaves the rest of the weights as they are initialized.

        Args:
            old_model_or_state: state dict of an FC1 instance.
            init_std: standard deviation for normal init of new weights.
        """
        # Sizes
        old_h = old_state['h1.weight'].shape[0]
        new_h = self.hidden_dim

        # Ensure new_h >= old_h
        if new_h < old_h:
            raise ValueError(f"New hidden_dim ({new_h}) must be >= old hidden_dim ({old_h})")

        with torch.no_grad():
            # 1) Copy and init h1 weights
            #   - copy the first old_h rows
            self.h1.weight.data[:old_h, :].copy_(old_state['h1.weight'])
            self.h1.bias.data[:old_h].copy_(old_state['h1.bias'])
            
            # Create masks for h1
            self.reused_h1_weight = torch.zeros_like(self.h1.weight.data, dtype=torch.bool)
            self.reused_h1_weight[:old_h, :] = True
            self.reused_h1_bias = torch.zeros_like(self.h1.bias.data, dtype=torch.bool)
            self.reused_h1_bias[:old_h] = True

            # 2) Copy and init out weights
            #   - copy the first old_h columns
            self.out.weight.data[:, :old_h].copy_(old_state['out.weight'])
            self.out.bias.data.copy_(old_state['out.bias'])
            
            # Create masks for out
            self.reused_out_weight = torch.zeros_like(self.out.weight.data, dtype=torch.bool)
            self.reused_out_weight[:, :old_h] = True
            self.reused_out_bias = torch.ones_like(self.out.bias.data, dtype=torch.bool) # All bias for output is reused
            
    def reuse_weights_and_freeze(self, old_state: dict):
        """
        Load weights from a smaller FC1 model state_dict into this wider model,
        and **freeze** the loaded weights (prevent them from training) using
        gradient hooks.

        Args:
            old_state: state dict of a smaller FC1 instance.
        """

        self.reuse_weights(old_state)

        old_h = old_state['h1.weight'].shape[0] # Get old_h again

        # --- 2. Freeze the reused parts using gradient hooks ---
        # Remove any previously attached freezing hooks to avoid duplicates
        self.remove_freeze_hooks()

        # Ensure the parameters require gradients globally first
        # (they should by default, but this makes the intent clear)
        self.h1.weight.requires_grad = True
        self.h1.bias.requires_grad = True
        self.out.weight.requires_grad = True
        self.out.bias.requires_grad = True

        # Define hook functions (closures that capture 'old_h')
        def h1_weight_hook(grad):
            # Zero out gradient for the reused part (first old_h rows)
            if grad is not None:
                grad_clone = grad.clone()
                grad_clone[:old_h, :] = 0
                return grad_clone
            return grad

        def h1_bias_hook(grad):
            # Zero out gradient for the reused part (first old_h elements)
             if grad is not None:
                grad_clone = grad.clone()
                grad_clone[:old_h] = 0
                return grad_clone
             return grad

        def out_weight_hook(grad):
            # Zero out gradient for the reused part (first old_h columns)
             if grad is not None:
                grad_clone = grad.clone()
                grad_clone[:, :old_h] = 0
                return grad_clone
             return grad

        def out_bias_hook(grad):
             # Zero out gradient for the entire output bias, as it was
             # fully copied from the old model in reuse_weights
             if grad is not None:
                grad_clone = grad.clone()
                grad_clone[:] = 0 # Or grad_clone.zero_()
                return grad_clone
             return grad

        # Register the hooks and store their handles
        h1_w_handle = self.h1.weight.register_hook(h1_weight_hook)
        self._freeze_handles.append(h1_w_handle)

        h1_b_handle = self.h1.bias.register_hook(h1_bias_hook)
        self._freeze_handles.append(h1_b_handle)

        out_w_handle = self.out.weight.register_hook(out_weight_hook)
        self._freeze_handles.append(out_w_handle)

        # Freeze the *entire* output bias, consistent with reuse_weights copying the whole thing
        out_b_handle = self.out.bias.register_hook(out_bias_hook)
        self._freeze_handles.append(out_b_handle)

        print(f"Reused weights from state_dict and registered hooks to freeze the first {old_h} hidden units' parameters.")


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
    
        stats = {}
        # Statistics for h1 weights
        if hasattr(self, 'reused_h1_weight'):
            reused_h1_weights = self.h1.weight.data[self.reused_h1_weight]
            non_reused_h1_weights = self.h1.weight.data[~self.reused_h1_weight]
            if reused_h1_weights.numel() > 0:
                stats['h1.weight_reused_mean'] = reused_h1_weights.mean().item()
                stats['h1.weight_reused_std'] = reused_h1_weights.std().item()
            if non_reused_h1_weights.numel() > 0:
                stats['h1.weight_non_reused_mean'] = non_reused_h1_weights.mean().item()
                stats['h1.weight_non_reused_std'] = non_reused_h1_weights.std().item()


        # Statistics for h1 bias
        if hasattr(self, 'reused_h1_bias'):
            reused_h1_bias = self.h1.bias.data[self.reused_h1_bias]
            non_reused_h1_bias = self.h1.bias.data[~self.reused_h1_bias]
            if reused_h1_bias.numel() > 0:
                stats['h1.bias_reused_mean'] = reused_h1_bias.mean().item()
                stats['h1.bias_reused_std'] = reused_h1_bias.std().item()
            if non_reused_h1_bias.numel() > 0:
                stats['h1.bias_non_reused_mean'] = non_reused_h1_bias.mean().item()
                stats['h1.bias_non_reused_std'] = non_reused_h1_bias.std().item()


        # Statistics for out weights
        if hasattr(self, 'reused_out_weight'):
            reused_out_weights = self.out.weight.data[self.reused_out_weight]
            non_reused_out_weights = self.out.weight.data[~self.reused_out_weight]
            if reused_out_weights.numel() > 0:
                stats['out.weight_reused_mean'] = reused_out_weights.mean().item()
                stats['out.weight_reused_std'] = reused_out_weights.std().item()
            if non_reused_out_weights.numel() > 0:
                stats['out.weight_non_reused_mean'] = non_reused_out_weights.mean().item()
                stats['out.weight_non_reused_std'] = non_reused_out_weights.std().item()


        # Statistics for out bias
        if hasattr(self, 'reused_out_bias'):
            reused_out_bias = self.out.bias.data[self.reused_out_bias]
            non_reused_out_bias = self.out.bias.data[~self.reused_out_bias]
            if reused_out_bias.numel() > 0:
                stats['out.bias_reused_mean'] = reused_out_bias.mean().item()
                stats['out.bias_reused_std'] = reused_out_bias.std().item()
            if non_reused_out_bias.numel() > 0:
                stats['out.bias_non_reused_mean'] = non_reused_out_bias.mean().item()
                stats['out.bias_non_reused_std'] = non_reused_out_bias.std().item()

        return stats



    

    
    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)