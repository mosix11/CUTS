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
    def __init__(
        self,
        dataset: Dataset,
        dataset_name:str=None,
        noise_type:str = 'symmetric', # between 'symmetric', 'asymmetric', and 'constant'
        noise_rate: float = 0.2,
        num_classes:int = None,
        target_class:int = None, # Only needed for 'constant' noise
        available_labels: list = None,
        seed=None,
        generator=None
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        
        if available_labels is None:
            self.available_labels = list(range(num_classes))
        else:
            self.available_labels = available_labels
        
        self.num_classes = num_classes 
        
        if noise_type == 'constant':
            if target_class is not None:
                self.target_class = target_class
            else:
                raise ValueError('For constant noise, the target class should be specified!')
        
        if seed and not generator:
            generator = torch.Generator().manual_seed(seed)
        elif seed and generator:
            generator = generator.manual_seed(seed)
        
        self.seed = seed
        self.generator = generator
            
        self.noisy_labels = None
        self.is_noisy_flags = None
        
        self.return_clean_labels = False

        if noise_rate > 0.0:
            self._add_noise_to_labels()
            
    def switch_to_clean_lables(self):
        self.return_clean_labels = True
    
    def switch_to_noisy_lables(self):
        self.return_clean_labels = False
        
    def replace_labels(self, new_labels):
        self.noisy_labels = new_labels
            
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
            # Ensure base_labels is a tensor
            if isinstance(base_dataset.targets, list):
                 base_labels = torch.tensor(base_dataset.targets, dtype=torch.long)
            else: # Assumes it's already a tensor or numpy array
                 base_labels = torch.as_tensor(base_dataset.targets, dtype=torch.long)

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
        self.original_labels = original_labels
        noisy_labels = original_labels.clone()
        self.is_noisy_flags = torch.zeros(len(original_labels))

        if self.noise_type == 'symmetric':
            # This already uses the fixed-count method on the whole dataset
            num_noisy_labels = int(self.noise_rate * len(original_labels))
            noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:num_noisy_labels]

            for idx in noisy_indices:
                current_label = original_labels[idx].item()
                possible_flips = [l for l in self.available_labels if l != current_label]
                
                if possible_flips:
                    rand_idx = torch.randint(0, len(possible_flips), (1,), generator=self.generator).item()
                    noisy_labels[idx] = possible_flips[rand_idx]
                    self.is_noisy_flags[idx] = 1.0
        elif self.noise_type == 'constant':
            # how many labels to corrupt
            num_noisy = int(self.noise_rate * len(original_labels))
            if num_noisy > 0:
                # pick that many random indices
                noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:num_noisy]
                # set them all to the single target class
                noisy_labels[noisy_indices] = self.target_class
                # mark them as noisy
                self.is_noisy_flags[noisy_indices] = 1.0
                
        elif self.noise_type == 'asymmetric':
            # This now uses the fixed-count method on a per-class basis
            if self.dataset_name is None:
                raise ValueError("To inject asymmetric noise, you must specify the dataset_name.")
            
            # Sub-classing with asymmetric noise is tricky, so we forbid it for now.
            if self.dataset_name == 'MNIST' and len(self.available_labels) != 10:
                raise RuntimeError("Asymmetric noise for sub-classed MNIST is not supported.")
            elif self.dataset_name == 'CIFAR10' and len(self.available_labels) != 10:
                raise RuntimeError("Asymmetric noise for sub-classed CIFAR-10 is not supported.")
            elif self.dataset_name == 'CIFAR100' and len(self.available_labels) != 100:
                raise RuntimeError("Asymmetric noise for sub-classed CIFAR-100 is not supported.")

            noise_map = {}
            if self.dataset_name == 'MNIST':
                noise_map = {7: 1, 2: 7, 5: 6, 6: 5, 3: 8}
            elif self.dataset_name == 'CIFAR10':
                noise_map = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
            elif self.dataset_name == 'CIFAR100':
                coarse_labels = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
                super_class_map = [[] for _ in range(20)]
                for i, c in enumerate(coarse_labels):
                    super_class_map[c].append(i)
                for super_class in super_class_map:
                    for i in range(len(super_class)):
                        source_label = super_class[i]
                        target_label = super_class[(i + 1) % len(super_class)]
                        noise_map[source_label] = target_label
            else:
                 raise ValueError(f"Asymmetric noise not implemented for dataset '{self.dataset_name}'.")

            for source_label, target_label in noise_map.items():
                # 1. Find all indices for the current source class
                class_indices = (original_labels == source_label).nonzero(as_tuple=True)[0]
                
                if len(class_indices) == 0:
                    continue

                # 2. Calculate the exact number of samples to flip
                num_to_flip = int(self.noise_rate * len(class_indices))
                if num_to_flip > 0:
                    # 3. Randomly select the indices to flip using a permutation
                    perm = torch.randperm(len(class_indices), generator=self.generator)
                    indices_to_flip_in_class = class_indices[perm[:num_to_flip]]
                    
                    # 4. Apply the noise and set the flags
                    noisy_labels[indices_to_flip_in_class] = target_label
                    self.is_noisy_flags[indices_to_flip_in_class] = 1.0

        self.noisy_labels = noisy_labels.long()

    def __getitem__(self, idx):
        # Retrieve the data from the wrapped dataset. The original label is ignored here.
        data, _ = self.dataset[idx] 
        
        # If noise was never added (noise_rate=0), return the original label from the dataset.
        if self.noisy_labels is None:
            return data, self.original_labels[idx], torch.tensor(False, dtype=torch.bool)
        
        # Return clean labels if requested
        if self.return_clean_labels:
            return data, self.original_labels[idx], torch.tensor(False, dtype=torch.bool)
        else:
            # Return the potentially noisy label and a flag indicating if it was corrupted.
            noisy_label = self.noisy_labels[idx]
            is_noisy = self.is_noisy_flags[idx] 
            return data, noisy_label, is_noisy

# class NoisyClassificationDataset(Dataset):
#     def __init__(self, dataset: Dataset, dataset_name:str=None, noise_type='symmetric', noise_rate=0.2, num_classes=10, available_labels: list = None, seed=None, generator=None):
#         super().__init__()
#         self.dataset = dataset
#         self.dataset_name = dataset_name
#         self.noise_type = noise_type
#         self.noise_rate = noise_rate
        
#         if available_labels is None:
#             self.available_labels = list(range(num_classes))
#         else:
#             self.available_labels = available_labels
        
#         # num_classes can still be useful for other potential logic, 
#         # but available_labels determines the noise flipping candidates.
#         self.num_classes = num_classes 
        
#         if seed and not generator:
#             generator = torch.Generator().manual_seed(seed)
#         elif seed and generator:
#             generator = generator.manual_seed(seed)
        
#         self.seed = seed
#         self.generator = generator
            
#         self.noisy_labels = None
#         self.is_noisy_flags = None
        
#         self.return_clean_labels = False

#         if noise_rate > 0.0:
#             self._add_noise_to_labels()
            
            
#     def switch_to_clean_lables(self):
#         self.return_clean_labels = True
    
#     def switch_to_noisy_lables(self):
#         self.return_clean_labels = False
        
#     def replace_labels(self, new_labels):
#         self.noisy_labels = new_labels
            
#     def __len__(self):
#         return len(self.dataset)

#     def _get_original_labels(self):
#         """
#         Traverses wrapped datasets (Subset) to get the original labels
#         from the base dataset, correctly composing indices from Subsets.
#         """
#         current_dataset = self.dataset
#         indices_chain = []

#         while isinstance(current_dataset, Subset):
#             indices_chain.append(current_dataset.indices)
#             current_dataset = current_dataset.dataset
            
#         base_dataset = current_dataset

#         if hasattr(base_dataset, 'targets'):
#             base_labels = torch.tensor(base_dataset.targets, dtype=torch.long)
#         else:
#             warnings.warn("Base dataset has no .targets attribute. Extracting labels by iterating, which can be slow.")
#             base_labels = torch.tensor([label for _, label in base_dataset], dtype=torch.long)

#         if not indices_chain:
#             return base_labels

#         indices_chain.reverse()
#         composed_indices = indices_chain[0]
#         for i in range(1, len(indices_chain)):
#             next_level_indices = indices_chain[i]
#             composed_indices = [composed_indices[j] for j in next_level_indices]
        
#         original_labels = base_labels[torch.tensor(composed_indices)]
        
#         return original_labels

#     def _add_noise_to_labels(self):
#         original_labels = self._get_original_labels()
#         self.original_labels = original_labels
#         noisy_labels = original_labels.clone()
#         self.is_noisy_flags = torch.zeros(len(original_labels))

#         if self.noise_type == 'symmetric':
#             num_noisy_labels = int(self.noise_rate * len(original_labels))
#             num_noisy_labels = min(num_noisy_labels, len(original_labels))
            
#             noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:num_noisy_labels]

#             for idx in noisy_indices:
#                 current_label = original_labels[idx].item()
                
#                 # Create a list of possible flips only from the available labels.
#                 possible_flips = [l for l in self.available_labels if l != current_label]
                
#                 if possible_flips:
#                     rand_idx = torch.randint(0, len(possible_flips), (1,), generator=self.generator).item()
#                     noisy_labels[idx] = possible_flips[rand_idx]
#                     self.is_noisy_flags[idx] = 1.0

#         elif self.noise_type == 'asymmetric':
#             if self.dataset_name == None:
#                 raise ValueError("In order to inject asymmetric noise, you have to specify the dataset name.")
#             if self.dataset_name == 'MNIST' and len(self.available_labels) != 10:
#                 raise RuntimeError("Assymetric noise for subclassed MNIST is not implemented")
#             elif self.dataset_name == 'CIFAR10' and len(self.available_labels) != 10:
#                 raise RuntimeError("Assymetric noise for subclassed CIFAR10 is not implemented")
#             elif self.dataset_name == 'CIFAR100' and len(self.available_labels) != 100:
#                 raise RuntimeError("Assymetric noise for subclassed CIFAR100 is not implemented")
#             for i, label_tensor in enumerate(original_labels):
#                 if torch.rand(1, generator=self.generator).item() < self.noise_rate:
                    
#                     label = label_tensor.item()
#                     flipped = False
                    
#                     if self.dataset_name == 'MNIST':
#                         pass
#                     elif self.dataset_name == 'CIFAR10':
#                         pass
#                     elif self.dataset_name == 'CIFAR100':
#                         pass
#                     # elif self.dataset_name == 'SVHN':
#                     #     pass
#                     # elif self.dataset_name == 'MoGSynthetic':
#                     #     pass
                    
                    
                    
#                     # # Assuming 3 is 'cat', 5 is 'dog'
#                     # if label == 3: 
#                     #     target_label = 5

#                     #     if target_label not in self.available_labels:
#                     #         raise ValueError(f"Asymmetric noise rule (3 -> 5) is invalid. Target label {target_label} is not in available_labels: {self.available_labels}")
#                     #     noisy_labels[i] = target_label
#                     #     flipped = True
#                     # # Assuming 2 is 'bird', 0 is 'airplane'
#                     # elif label == 2: 
#                     #     target_label = 0
#                     #     # *** ADDED VALIDATION ***
#                     #     if target_label not in self.available_labels:
#                     #         raise ValueError(f"Asymmetric noise rule (2 -> 0) is invalid. Target label {target_label} is not in available_labels: {self.available_labels}")
#                     #     noisy_labels[i] = target_label
#                     #     flipped = True
#                     # # Add more specific rules as needed

#                     if flipped:
#                         self.is_noisy_flags[i] = 1.0

#         elif self.noise_type == 'instance_dependent':
#             raise NotImplementedError("Instance-dependent noise is complex and typically requires a separate model.")

#         self.noisy_labels = noisy_labels.long()

#     def __getitem__(self, idx):
#         data, _ = self.dataset[idx] 
        
#         if self.noisy_labels is None:
#             _, original_label = self.dataset[idx]
#             return data, original_label, torch.tensor(False, dtype=torch.bool)
        
        
#         if self.return_clean_labels:
#             original_label = self.original_labels[idx]
#             return data, original_label, torch.tensor(False, dtype=torch.bool)
#         else:
#             noisy_label = self.noisy_labels[idx]
#             is_noisy = self.is_noisy_flags[idx] 
#             return data, noisy_label, is_noisy



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

