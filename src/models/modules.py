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
        self.initialized_count = 0
        self.initialization_done = False

    def forward(self, X, indices):
        if self.p_fixed == 1.0:
            return X

        B, C, H, W = X.shape
        device = X.device
        fixed_channels = int(self.p_fixed * C)

        if self.training:
            if self.mask_tensor is None:
                self.mask_tensor = torch.zeros((self.max_id, C, H, W), dtype=torch.bool, device=device)

            if not self.initialization_done:
                for i in range(B):
                    sample_idx = indices[i].item()

                    if torch.count_nonzero(self.mask_tensor[sample_idx]) != 0:
                        continue  # Already initialized

                    mask = torch.zeros((C, H, W), device=device)
                    mask[:fixed_channels] = 1.0

                    mem_channels = C - fixed_channels
                    gen = torch.Generator(device='cpu').manual_seed(sample_idx)
                    mem_mask = torch.bernoulli(torch.full((mem_channels,), self.p_mem), generator=gen).to(device)
                    mem_mask = mem_mask.view(-1, 1, 1).expand(-1, H, W)
                    mask[fixed_channels:] = mem_mask

                    self.mask_tensor[sample_idx] = mask
                    self.initialized_count += 1

                if self.initialized_count >= self.max_id:
                    self.initialization_done = True  # No more loops in future epochs

            masks = self.mask_tensor[indices].to(device)
            return X * masks

        else:
            X_fixed = X[:, :fixed_channels]
            X_mem = X[:, fixed_channels:] * self.p_mem
            return torch.cat([X_fixed, X_mem], dim=1)