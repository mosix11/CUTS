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
        key = y.item() if isinstance(y, torch.Tensor) else y
        return x, self.map[key]

        
class NoisyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, is_noisy_applied: bool = False):
        super().__init__()
        self.base_dataset = base_dataset
        self.is_noisy_applied = is_noisy_applied

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        is_noisy_flag = torch.tensor(False, dtype=torch.bool)

        if self.is_noisy_applied:
            current_dataset = self.base_dataset
            original_idx = idx

            # Traverse all wrappers to find the base dataset and the final original index
            while isinstance(current_dataset, (Subset, LabelRemapper)):
                if isinstance(current_dataset, LabelRemapper):
                    current_dataset = current_dataset.base
                elif isinstance(current_dataset, Subset):
                    # Map the current index through the subset's indices
                    original_idx = current_dataset.indices[original_idx]
                    current_dataset = current_dataset.dataset

            # Now current_dataset should be the original dataset (e.g., CIFAR10)
            if hasattr(current_dataset, 'is_noisy'):
                is_noisy_flag = current_dataset.is_noisy[original_idx]
            else:
                print("Warning: is_noisy attribute not found on the base dataset.")

        return x, y, is_noisy_flag

    def __len__(self):
        return len(self.base_dataset)

def apply_label_noise(dataset, label_noise, class_subset, generator=None):
    current_dataset = dataset
    indices_chain = []

    # Traverse wrappers to find the base dataset, collecting indices from each Subset
    while isinstance(current_dataset, (Subset, LabelRemapper)):
        if isinstance(current_dataset, LabelRemapper):
            current_dataset = current_dataset.base
        elif isinstance(current_dataset, Subset):
            indices_chain.append(current_dataset.indices)
            current_dataset = current_dataset.dataset

    base_dataset = current_dataset

    # If there were Subset wrappers, compose their indices to get the final list
    if indices_chain:
        # The collected chain is from the outermost to the innermost Subset.
        # We need to reverse it to compose them correctly.
        indices_chain.reverse()
        
        # Start with the indices that map to the base_dataset
        original_indices = indices_chain[0]
        
        # Sequentially apply the mappings from the other Subsets
        for i in range(1, len(indices_chain)):
            next_level_indices = indices_chain[i]
            original_indices = [original_indices[j] for j in next_level_indices]
    else:
        # If no Subset was found, the indices are simply the range of the dataset length
        original_indices = list(range(len(dataset)))

    # Ensure the base dataset's targets are a tensor for advanced indexing
    if not isinstance(base_dataset.targets, torch.Tensor):
        base_dataset.targets = torch.tensor(base_dataset.targets)
    
    num_samples = len(dataset)
    
    # Get the original labels for the samples in the current dataset view
    original_labels = base_dataset.targets[original_indices].clone().detach()

    # Define the pool of valid labels for flipping
    if class_subset and len(class_subset) > 0:
        allowed_labels = torch.tensor(class_subset, device=original_labels.device)
    else:
        num_classes = 10
        if hasattr(base_dataset, 'classes'):
            num_classes = len(base_dataset.classes)
        allowed_labels = torch.arange(num_classes, device=original_labels.device)
    
    num_allowed_classes = len(allowed_labels)
    num_to_flip = int(num_samples * label_noise)

    # Select random indices to flip within the current dataset view
    perm = torch.randperm(num_samples, generator=generator)
    flip_indices_relative = perm[:num_to_flip]
    
    noise_mask = torch.zeros(num_samples, dtype=torch.bool)
    noise_mask[flip_indices_relative] = True

    labels_to_flip = original_labels[flip_indices_relative]

    # Generate new random labels, ensuring they are different from the original
    random_labels = torch.randint(0, num_allowed_classes, (num_to_flip,), generator=generator)
    new_labels = allowed_labels[random_labels]

    conflict_mask = (new_labels == labels_to_flip)
    while conflict_mask.any():
        num_conflicts = conflict_mask.sum()
        new_random_indices = torch.randint(0, num_allowed_classes, (num_conflicts,), generator=generator)
        new_labels[conflict_mask] = allowed_labels[new_random_indices]
        conflict_mask = (new_labels == labels_to_flip)

    noisy_labels = original_labels.clone()
    noisy_labels[flip_indices_relative] = new_labels

    # Update the targets in the base_dataset using the final mapped indices
    base_dataset.targets[original_indices] = noisy_labels

    # Initialize or update the is_noisy flag on the base_dataset
    if not hasattr(base_dataset, 'is_noisy'):
        base_dataset.is_noisy = torch.zeros(len(base_dataset.targets), dtype=torch.bool)
    
    # Create a temporary mask for the original_indices and apply the noise mask
    temp_is_noisy_mask = base_dataset.is_noisy[original_indices]
    temp_is_noisy_mask[flip_indices_relative] = True
    base_dataset.is_noisy[original_indices] = temp_is_noisy_mask

    return dataset



