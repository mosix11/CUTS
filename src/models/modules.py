import torch
import torch.nn as nn
import torch.nn.functional as F

class ExampleTiedDropout(nn.Module):
    """
    Directs memorization to specific neurons using example-specific dropout masks.
    - A fixed set of neurons (generalization neurons) are always active.
    - The remaining neurons (memorization neurons) are randomly activated per-example,
      but with a fixed seed based on the example index.
    """

    def __init__(self, p_gen=0.2, p_mem=0.1, num_training_samples=60000):
        super().__init__()
        self.p_fixed = p_gen
        self.p_mem = p_mem
        self.max_id = num_training_samples
        self.mask_tensor = None  # shape: [max_id, C, H, W]

    def forward(self, X, indices):
        # Skip dropout entirely if p_fixed is 1.0 (i.e., keep everything)
        if self.p_fixed == 1.0:
            return X

        B, C, H, W = X.shape
        device = X.device
        fixed_channels = int(self.p_fixed * C)

        
        if self.training:
            indices = indices.to('cpu')
            # Create mask_tensor on first forward call
            if self.mask_tensor is None:
                self.mask_tensor = torch.zeros((self.max_id, C, H, W), dtype=torch.bool, device='cpu')
            
            # Build per-example dropout masks
            for i in range(B):
                sample_idx = indices[i].cpu().item()

                # Skip mask generation if already exists
                if torch.count_nonzero(self.mask_tensor[sample_idx]) != 0:
                    continue

                # Start with fixed neurons
                mask = torch.zeros((C, H, W), device='cpu')
                mask[:fixed_channels] = 1.0

                # Generate example-specific memorization mask
                mem_channels = C - fixed_channels
                gen = torch.Generator(device='cpu').manual_seed(sample_idx)
                mem_mask = torch.bernoulli(torch.full((mem_channels,), self.p_mem), generator=gen).to('cpu')
                mem_mask = mem_mask.view(-1, 1, 1).expand(-1, H, W)
                mask[fixed_channels:] = mem_mask

                # Store the per-example mask
                self.mask_tensor[sample_idx] = mask

            # Apply the dropout mask
            masks = self.mask_tensor[indices].to(device)  # shape [B, C, H, W]
            return X * masks

        else:
            # In eval mode, scale memorization neurons by p_mem
            X_fixed = X[:, :fixed_channels]
            X_mem = X[:, fixed_channels:] * self.p_mem
            return torch.cat([X_fixed, X_mem], dim=1)