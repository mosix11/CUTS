import torch
from torch.utils.data import Dataset, Subset

class LabelRemapper(Dataset):
        """
        Wraps any dataset whose __getitem__ returns (x, y)
        and remaps y via a provided dict mapping_orig2new.
        """
        def __init__(self, base_dataset: Dataset, mapping_orig2new: dict):
            super().__init__()
            self.base = base_dataset
            self.map = mapping_orig2new

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y = self.base[idx]
            return x, self.map[y]
        
        
        
        
class NoisyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, is_noisy_applied: bool = False):
        super().__init__()
        self.base_dataset = base_dataset
        self.is_noisy_applied = is_noisy_applied # Rename is_noisy to is_noisy_applied to avoid confusion with the sample-level flag

    def __getitem__(self, idx):
        # Get the item from the wrapped dataset. This will handle label remapping if LabelRemapper is in the chain.
        x, y = self.base_dataset[idx]

        is_noisy_flag = torch.tensor(False, dtype=torch.bool)

        if self.is_noisy_applied:
            # Traverse the dataset wrappers to find the base dataset with `is_noisy`
            current_dataset = self.base_dataset
            original_idx = idx

            # Handle LabelRemapper
            if isinstance(current_dataset, LabelRemapper):
                current_dataset = current_dataset.base # Unwrap LabelRemapper

            # Handle Subset
            if isinstance(current_dataset, Subset):
                original_idx = current_dataset.indices[idx]
                current_dataset = current_dataset.dataset # Unwrap Subset

            # Now current_dataset should be the original dataset (e.g., MNIST)
            if hasattr(current_dataset, 'is_noisy'):
                is_noisy_flag = current_dataset.is_noisy[original_idx]
            else:
                # This case should ideally not happen if _apply_label_noise was called
                print("Warning: is_noisy attribute not found on the base dataset.")

        return x, y, is_noisy_flag

    def __len__(self):
        return len(self.base_dataset)