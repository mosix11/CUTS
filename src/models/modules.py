import torch
import torch.nn as nn
import torch.nn.functional as F

# class ExampleTiedDropout(nn.Module):
#     """
#     Directs memorization to specific neurons using example-specific dropout masks.
#     - A fixed set of neurons (generalization neurons) are always active.
#     - The remaining neurons (memorization neurons) are randomly activated per-example,
#       but with a fixed seed based on the example index.
#     """

#     def __init__(self, p_gen=0.2, p_mem=0.1, num_training_samples=60000):
#         super().__init__()
#         self.p_fixed = p_gen
#         self.p_mem = p_mem
#         self.max_id = num_training_samples
#         self.mask_tensor = None  # shape: [max_id, C, H, W]
#         self.initialized_count = 0
#         self.initialization_done = False

#     def forward(self, X, indices):
#         if self.p_fixed == 1.0:
#             return X

#         B, C, H, W = X.shape
#         device = X.device
#         fixed_channels = int(self.p_fixed * C)
#         if self.training:
#             if self.mask_tensor is None:
#                 self.mask_tensor = torch.zeros((self.max_id, C, H, W), dtype=torch.bool, device=device)

#             if not self.initialization_done:
#                 for i in range(B):
#                     sample_idx = indices[i].item()

#                     if torch.count_nonzero(self.mask_tensor[sample_idx]) != 0:
#                         continue  # Already initialized

#                     mask = torch.zeros((C, H, W), device=device)
#                     mask[:fixed_channels] = 1.0

#                     mem_channels = C - fixed_channels
#                     gen = torch.Generator(device='cpu').manual_seed(sample_idx)
                    
#                     # TODO
#                     # Maybe we can generate the binary mask of the memory channels in the init function since
#                     # We know the number of the samples.
                    
#                     mem_mask = torch.bernoulli(torch.full((mem_channels,), self.p_mem), generator=gen).to(device)
#                     mem_mask = mem_mask.view(-1, 1, 1).expand(-1, H, W)
#                     mask[fixed_channels:] = mem_mask

#                     self.mask_tensor[sample_idx] = mask
#                     self.initialized_count += 1

#                 if self.initialized_count >= self.max_id:
#                     self.initialization_done = True  # No more loops in future epochs

#             masks = self.mask_tensor[indices].to(device)
#             return X * masks

#         else:
#             X_fixed = X[:, :fixed_channels]
#             X_mem = X[:, fixed_channels:] * self.p_mem
#             return torch.cat([X_fixed, X_mem], dim=1)


import torch
import torch.nn as nn

class ExampleTiedDropout(nn.Module):
    """
    Implements Example-Tied Dropout.

    This layer localizes memorization by tying a specific dropout mask to each training example.
    It divides channels into "fixed" (for generalization) and "memorization" sets.

    Improvements based on user feedback:
    1.  **Memory-Efficient Masks:** Stores only 1D boolean vectors for masks and generates
        full masks on the fly, dramatically reducing memory usage.
    2.  **Stateful & Self-Contained:** No longer requires the 'epoch' argument. It internally
        tracks which masks have been generated.
    3.  **Safe Randomness:** Uses a torch.Generator for isolated and reproducible
        random mask creation without global side effects.
    4.  **Standard PyTorch Practice:** Uses `self.training` (toggled by `model.train()`
        and `model.eval()`) to control its behavior.
    5.  **Flexible Evaluation:** An `eval_mode` argument ('standard' or 'drop') controls
        behavior during evaluation.
    """
    def __init__(self, num_training_samples: int, p_fixed: float = 0.2, p_mem: float = 0.1, eval_mode: str = "standard"):
        super().__init__()
        if not 0.0 <= p_fixed <= 1.0:
            raise ValueError(f"p_fixed must be between 0 and 1, but got {p_fixed}")
        if not 0.0 <= p_mem <= 1.0:
            raise ValueError(f"p_mem must be between 0 and 1, but got {p_mem}")
        if eval_mode not in ["standard", "drop"]:
            raise ValueError(f"eval_mode must be 'standard' or 'drop', but got {eval_mode}")

        self.num_training_samples = num_training_samples
        self.p_fixed = p_fixed
        self.p_mem = p_mem
        self.eval_mode = eval_mode

        self.masks = None
        self.masks_initialized = None
        
        self.generator = None

    def forward(self, X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.p_fixed == 1.0:
            return X # No dropout to apply

        

        if self.training:
            
            device = X.device
        
            # Lazily initialize buffers on the first forward pass
            if self.masks is None:
                num_channels = X.shape[1]
                # Stores the 1D boolean mask for each training sample
                self.masks = torch.zeros(self.num_training_samples, num_channels, dtype=torch.bool, device=device)
                # Tracks which masks have been created
                self.masks_initialized = torch.zeros(self.num_training_samples, dtype=torch.bool, device=device)
                
                self.generator = torch.Generator(device=device)
                
            # Determine which samples in the current batch need their masks generated for the first time
            uninitialized_indices = idx[~self.masks_initialized[idx]]

            if len(uninitialized_indices) > 0:
                with torch.no_grad():
                    num_channels = self.masks.shape[1]
                    num_fixed = int(self.p_fixed * num_channels)
                    
                    # Generate a new mask for each uninitialized sample
                    for i in uninitialized_indices:
                        self.generator.manual_seed(i.item())
                        
                        # Fixed neurons are always kept (mask = True)
                        fixed_mask = torch.ones(num_fixed, dtype=torch.bool, device=device)
                        
                        # Memorization neurons are kept with probability p_mem
                        num_mem = num_channels - num_fixed
                        
                        # Create the tensor of probabilities first (without the generator)
                        probs = torch.full((num_mem,), self.p_mem, device=device)
                        # Then pass the generator to torch.bernoulli
                        mem_mask = torch.bernoulli(probs, generator=self.generator).to(torch.bool)
                        
                        self.masks[i] = torch.cat((fixed_mask, mem_mask))
                    
                    self.masks_initialized[uninitialized_indices] = True

            # Retrieve the 1D masks for the current batch
            batch_mask_1d = self.masks[idx]

            # Expand the 1D channel mask to the full shape of the input tensor on the fly
            # This is extremely memory-efficient.
            # Shape goes from (B, C) -> (B, C, 1, 1) to allow broadcasting over H and W
            mask = batch_mask_1d.view(*batch_mask_1d.shape, 1, 1).expand_as(X)
            
            # Apply the mask
            return X * mask

        else: # Evaluation mode (self.training is False)
            num_channels = X.shape[1]
            num_fixed = int(self.p_fixed * num_channels)
            
            if self.eval_mode == "standard":
                # Standard dropout scaling: scale the memorization neurons by p_mem
                # The clone() is important to avoid in-place modification of a tensor that
                # might be needed elsewhere.
                output = X.clone()
                if num_fixed < num_channels: # Only apply scaling if there are mem neurons
                    output[:, num_fixed:] *= self.p_mem
                return output
            
            elif self.eval_mode == "drop":
                # Drop memorization neurons entirely and rescale fixed neurons
                output = torch.zeros_like(X)
                if num_fixed > 0:
                    # Scale to preserve the expected activation sum. If p_fixed is 0, this is NaN.
                    scale = (self.p_fixed + self.p_mem) / self.p_fixed
                    output[:, :num_fixed] = X[:, :num_fixed] * scale
                return output
                
        
    
    
    
    
# class ExampleTiedDropout(torch.nn.Module):
#     # this class is similar to batch tied dropout, 
#     # but instead of tying neurons in a batch, we tie neurons in a set of examples

#     def __init__(self, p_fixed = 0.2, p_mem = 0.1, eval_mode = "standard"):
#         super(ExampleTiedDropout, self).__init__()
#         self.seed = 101010
#         self.max_id = 50000
#         self.p_mem = p_mem
#         self.p_fixed = p_fixed
#         self.eval_mode = eval_mode
#         self.mask_tensor = None

#     def forward(self, X, idx):
#         if self.p_fixed == 1:
#             return X

#         if self.training:
#             # create a mask based on the index (idx)

#             mask = torch.zeros_like(X).cpu()
#             shape = X.shape[1]

#             if epoch > 0:
#                 #get mask from self.mask_tensor
#                 mask = self.mask_tensor[idx]

#             elif epoch == 0:
#                 #keep all neurons with index less than self.p_fixed*shape
#                 mask[:, :int(self.p_fixed*shape)] = 1

#                 # Fraction of elements to keep
#                 p_mem = self.p_mem

#                 # Generate a random mask for each row in the input tensor
#                 shape_of_mask = shape - int(self.p_fixed*shape)
#                 for i in range(X.shape[0]):
#                     torch.manual_seed(idx[i].item())
#                     curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
#                     #repeat curr_mask along dimension 2 and 3 to have the same shape as X
#                     curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1)
#                     mask[i][int(self.p_fixed*shape):] = curr_mask
#                     # In an initial implementation, rather than repeating the same mask along all dimensions,
#                     # the following was done. This meant that we will randomly have 0s and 1s along different dimensions
#                     # removed this so that it preserves image pixel semantics.

#                     # mask[i][int(self.p_fixed*shape):] = torch.bernoulli(torch.full((1, shape_of_mask, X.shape[2], X.shape[3]), p_mem))
                    

#                 if self.mask_tensor is None:
#                     self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2], X.shape[3])
#                 #assign mask at positions given by idx
#                 self.mask_tensor[idx] = mask
            
#             # Apply the mask to the input tensor
#             X = X * mask.cuda()


#         elif self.eval_mode == "standard":
#             #At test time we will renormalize outputs from the non-fixed neurons based on the number of neuron sets
#             #we will keep the fixed neurons unmodified
#             shape = X.shape[1]
#             X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
#             X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]*self.p_mem

#         elif self.eval_mode == "drop":
#             shape = X.shape[1]
#             X[:, int(self.p_fixed*shape):] = 0
#             X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed
        
#         return X