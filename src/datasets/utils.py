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
            return x, y, idx, is_noisy 
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



class BinarizedClassificationDataset(Dataset):
    
    def __init__(self, dataset: Dataset, target_class:int):
        super().__init__()
        self.dataset = dataset
        self.target_class = target_class
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            data = list(data)
        y = data[1]
        if y == self.target_class:
            data[1] = 1.0
        else:
            data[1] = 0.0
        return data

