import torch
import torch.nn as nn



class ExampleTiedDropout(nn.Module):
    def __init__(self, num_training_samples: int, num_channels: int, p_fixed: float = 0.2, p_mem: float = 0.1, eval_mode: str = "standard"):
        super().__init__()

        self.num_training_samples = num_training_samples
        self.num_channels = num_channels # Store num_channels
        self.p_fixed = p_fixed
        self.p_mem = p_mem
        self.eval_mode = eval_mode

        masks = torch.zeros(num_training_samples, num_channels, dtype=torch.bool)
        masks_initialized = torch.zeros(num_training_samples, dtype=torch.bool)
        self.register_buffer("masks", masks)
        self.register_buffer("masks_initialized", masks_initialized)
        
        # Generator can still be initialized lazily to ensure correct device
        self.generator = None

    def forward(self, X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # On the first forward pass, just initialize the generator on the correct device.
        if self.generator is None:
            self.generator = torch.Generator(device=X.device)

        if self.training:

            if self.p_fixed == 1.0:
                return X

            uninitialized_indices = idx[~self.masks_initialized[idx]]

            if len(uninitialized_indices) > 0:
                with torch.no_grad():
                    num_fixed = int(self.p_fixed * self.num_channels)
                    
                    for i in uninitialized_indices:
                        self.generator.manual_seed(i.item())
                        
                        fixed_mask = torch.ones(num_fixed, dtype=torch.bool, device=X.device)
                        
                        num_mem = self.num_channels - num_fixed
                        if num_mem > 0:
                            probs = torch.full((num_mem,), self.p_mem, device=X.device)
                            mem_mask = torch.bernoulli(probs, generator=self.generator).to(torch.bool)
                            self.masks[i] = torch.cat((fixed_mask, mem_mask))
                        else:
                            self.masks[i] = fixed_mask

                    self.masks_initialized[uninitialized_indices] = True

            batch_mask_1d = self.masks[idx]
            mask = batch_mask_1d.view(*batch_mask_1d.shape, 1, 1).expand_as(X)
            return X * mask

        else: # Evaluation mode
            # ... (evaluation logic remains exactly the same)
            num_fixed = int(self.p_fixed * self.num_channels)
            
            if self.eval_mode == "standard":
                output = X.clone()
                if num_fixed < self.num_channels:
                    output[:, num_fixed:] *= self.p_mem
                return output
            
            elif self.eval_mode == "drop":
                output = torch.zeros_like(X)
                if self.p_fixed > 0:
                    scale = (self.p_fixed + self.p_mem) / self.p_fixed
                    output[:, :num_fixed] = X[:, :num_fixed] * scale
                    # output[:, :num_fixed] = X[:, :num_fixed]
                return output
                
        return X