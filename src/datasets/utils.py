import torch
from torch.utils.data import Dataset, Subset
import warnings


class DatasetWithIndex(Dataset):
    
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            x, y, is_noisy = data
            return x, y, is_noisy, idx
        elif len(data) == 2:
            x, y = data
            return x, y, idx
        else:
            raise RuntimeError('Data structure unknown!')

class LabelRemapper(Dataset):
    """
    Wraps any dataset whose __getitem__ returns (x, y)
    and remaps y via a provided dict mapping_orig2new.
    """
    def __init__(self, dataset: Dataset, mapping_orig2new: dict):
        super().__init__()
        self.dataset = dataset
        self.map = mapping_orig2new

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            x, y, is_noisy = data
            key = y.item() if isinstance(y, torch.Tensor) else y
            return x, self.map[key], is_noisy
        elif len(data) == 2:
            x, y = data
            key = y.item() if isinstance(y, torch.Tensor) else y
            return x, self.map[key]
        else:
            raise RuntimeError('Data structure unknown!')



class NoisyClassificationDataset(Dataset):
    def __init__(self, dataset: Dataset, noise_type='symmetric', noise_rate=0.2, num_classes=10, available_labels: list = None, seed=None, generator=None):
        super().__init__()
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        
        if available_labels is None:
            self.available_labels = list(range(num_classes))
        else:
            self.available_labels = available_labels
        
        # num_classes can still be useful for other potential logic, 
        # but available_labels determines the noise flipping candidates.
        self.num_classes = num_classes 
        
        if seed and not generator:
            generator = torch.Generator().manual_seed(seed)
        elif seed and generator:
            generator = generator.manual_seed(seed)
        
        self.seed = seed
        self.generator = generator
            
        self.noisy_labels = None
        self.is_noisy_flags = None

        if noise_rate > 0.0:
            self._add_noise_to_labels()
            
    def __len__(self):
        return len(self.dataset)

    def _get_original_labels(self):
        """
        Traverses wrapped datasets (Subset) to get the original labels
        from the base dataset, correctly composing indices from Subsets.
        """
        current_dataset = self.dataset
        indices_chain = []

        while isinstance(current_dataset, Subset):
            indices_chain.append(current_dataset.indices)
            current_dataset = current_dataset.dataset
            
        base_dataset = current_dataset

        if hasattr(base_dataset, 'targets'):
            base_labels = torch.tensor(base_dataset.targets, dtype=torch.long)
        else:
            warnings.warn("Base dataset has no .targets attribute. Extracting labels by iterating, which can be slow.")
            base_labels = torch.tensor([label for _, label in base_dataset], dtype=torch.long)

        if not indices_chain:
            return base_labels

        indices_chain.reverse()
        composed_indices = indices_chain[0]
        for i in range(1, len(indices_chain)):
            next_level_indices = indices_chain[i]
            composed_indices = [composed_indices[j] for j in next_level_indices]
        
        original_labels = base_labels[torch.tensor(composed_indices)]
        
        return original_labels

    def _add_noise_to_labels(self):
        original_labels = self._get_original_labels()
        noisy_labels = original_labels.clone()
        self.is_noisy_flags = torch.zeros(len(original_labels))

        if self.noise_type == 'symmetric':
            num_noisy_labels = int(self.noise_rate * len(original_labels))
            num_noisy_labels = min(num_noisy_labels, len(original_labels))
            
            noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:num_noisy_labels]

            for idx in noisy_indices:
                current_label = original_labels[idx].item()
                
                # Create a list of possible flips only from the available labels.
                possible_flips = [l for l in self.available_labels if l != current_label]
                
                if possible_flips:
                    rand_idx = torch.randint(0, len(possible_flips), (1,), generator=self.generator).item()
                    noisy_labels[idx] = possible_flips[rand_idx]
                    self.is_noisy_flags[idx] = 1.0

        elif self.noise_type == 'asymmetric':
            for i, label_tensor in enumerate(original_labels):
                if torch.rand(1, generator=self.generator).item() < self.noise_rate:
                    label = label_tensor.item()
                    flipped = False
                    
                    # # Assuming 3 is 'cat', 5 is 'dog'
                    # if label == 3: 
                    #     target_label = 5

                    #     if target_label not in self.available_labels:
                    #         raise ValueError(f"Asymmetric noise rule (3 -> 5) is invalid. Target label {target_label} is not in available_labels: {self.available_labels}")
                    #     noisy_labels[i] = target_label
                    #     flipped = True
                    # # Assuming 2 is 'bird', 0 is 'airplane'
                    # elif label == 2: 
                    #     target_label = 0
                    #     # *** ADDED VALIDATION ***
                    #     if target_label not in self.available_labels:
                    #         raise ValueError(f"Asymmetric noise rule (2 -> 0) is invalid. Target label {target_label} is not in available_labels: {self.available_labels}")
                    #     noisy_labels[i] = target_label
                    #     flipped = True
                    # # Add more specific rules as needed

                    if flipped:
                        self.is_noisy_flags[i] = 1.0

        elif self.noise_type == 'instance_dependent':
            raise NotImplementedError("Instance-dependent noise is complex and typically requires a separate model.")

        self.noisy_labels = noisy_labels.long()

    def __getitem__(self, idx):
        data, _ = self.dataset[idx] 
        
        if self.noisy_labels is None:
            _, original_label = self.dataset[idx]
            return data, original_label, torch.tensor(False, dtype=torch.bool)
        
        noisy_label = self.noisy_labels[idx]
        is_noisy = self.is_noisy_flags[idx] 
        return data, noisy_label, is_noisy





def apply_label_noise(dataset, label_noise, class_subset, seed=None, generator=None):
    
    if seed and not generator:
        generator = torch.Generator().manual_seed(seed)
    elif seed and generator:
        generator = generator.manual_seed(seed)
    
    
    current_dataset = dataset
    indices_chain = []

    # Traverse wrappers to find the base dataset, collecting indices from each Subset
    while isinstance(current_dataset, (Subset, LabelRemapper)):
        if isinstance(current_dataset, Subset):
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



