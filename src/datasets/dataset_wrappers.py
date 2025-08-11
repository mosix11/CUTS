import torch
from torch.utils.data import Dataset, Subset
import warnings
import numpy as np

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
        T_mat: np.ndarray = None, # only for noise_type='T_matrix'
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
        elif noise_type == 'T_matrix':
            if T_mat is not None:
                self.T_mat = T_mat
            else:
                raise ValueError('For generating noise based on Transition Matrix, the Transition Matrix should be passed as `T_mat`.')
        
        
        if seed and not generator:
            generator = torch.Generator().manual_seed(seed)
        elif seed and generator:
            generator = generator.manual_seed(seed)
        
        self.seed = seed
        self.generator = generator
            
        self.noisy_labels = None
        self.is_noisy_flags = None
        
        self.return_clean_labels = False

        
        self._add_noise_to_labels()
            
            
    def switch_to_clean_lables(self):
        self.return_clean_labels = True
    
    def switch_to_noisy_lables(self):
        self.return_clean_labels = False
        
    def replace_labels(self, new_labels):
        self.noisy_labels = new_labels
        for idx, orig_lbl in enumerate(self.original_labels):
            if orig_lbl != new_labels[idx]:
                self.is_noisy_flags[idx] = 1.0
            else:
                self.is_noisy_flags[idx] = 0.0
    
    def get_original_labels(self):
        return self.original_labels
            
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

        if not (self.noise_rate > 0.0 and self.noise_rate <= 1.0):
            return
        
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
            elif self.dataset_name == 'Clothing1M' and len(self.available_labels) != 14:
                raise RuntimeError("Asymmetric noise for sub-classed Clothing1M is not supported.")

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
            elif self.dataset_name == 'Clothing1M':
                # noise_map = {
                #     0: 1,   # T-Shirt -> Shirt
                #     1: 0,   # Shirt -> T-Shirt
                #     2: 4,   # Knitwear -> Sweater
                #     4: 2,   # Sweater -> Knitwear
                #     3: 11,  # Chiffon -> Dress
                #     11: 3,  # Dress -> Chiffon
                #     6: 8,   # Windbreaker -> Downcoat
                #     8: 6,   # Downcoat -> Windbreaker
                #     7: 9,   # Jacket -> Suit
                #     9: 7,   # Suit -> Jacket
                #     5: 4,   # Hoodie -> Sweater
                #     10: 11, # Shawl -> Dress
                #     12: 7,  # Vest -> Jacket
                #     13: 0,  # Underwear -> T-Shirt
                # }
                noise_map = {
                    2: 4,   # Knitwear -> Sweater
                    4: 2,   # Sweater -> Kintwear
                    9: 6,   # Suit -> Windbreaker
                    12: 11, # Vest -> Dress
                    8: 12,  # Downcoat -> Vest
                    13: 12, # Underwear -> Vest
                    6: 8,   # Windbreaker -> Downcoat
                    10: 2   # Shawl -> Kintwear
                }
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
                    
        elif self.noise_type == 'T_matrix':
            if not isinstance(self.T_mat, np.ndarray):
                raise TypeError("T_mat must be a numpy.ndarray of shape (num_classes, num_classes).")
            K = self.num_classes
            if self.T_mat.shape != (K, K):
                raise ValueError(f"T_mat must have shape ({K}, {K}), got {self.T_mat.shape}.")

            T = self.T_mat  # do not modify
            # Tolerances for numerical checks
            tol_row = 1e-6
            tol_diag = 1e-9

            # 1) Rows sum to ~1
            row_sums = T.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=tol_row):
                raise ValueError(f"T_mat rows must sum to 1 (±{tol_row}). Got {row_sums}.")

            # 2) Zero diagonal (so flips always change the label)
            if not np.all(np.abs(np.diag(T)) <= tol_diag):
                raise ValueError(f"T_mat diagonal must be ~0 (≤{tol_diag}). Got {np.diag(T)}.")

            # 3) No negatives
            if (T < -tol_diag).any():
                raise ValueError("T_mat must be nonnegative.")

            # Convert to torch tensor for multinomial sampling (no renormalization)
            T_torch = torch.from_numpy(T).to(dtype=torch.float64)

            # ---- Apply fixed-count corruption per class (matches your asymmetric path) ----
            for cls in range(K):
                class_indices = (original_labels == cls).nonzero(as_tuple=True)[0]
                if len(class_indices) == 0:
                    continue

                num_to_flip = int(self.noise_rate * len(class_indices))
                if num_to_flip <= 0:
                    continue

                # Choose which samples of this class to corrupt (fixed-count)
                perm = torch.randperm(len(class_indices), generator=self.generator)
                idx_to_flip = class_indices[perm[:num_to_flip]]

                # Sample target labels from the row distribution T[cls], zero-diagonal ensures different label
                probs = T_torch[cls]  # shape: (K,)
                sampled_targets = torch.multinomial(
                    probs, num_samples=num_to_flip, replacement=True, generator=self.generator
                ).to(dtype=torch.long)

                noisy_labels[idx_to_flip] = sampled_targets
                self.is_noisy_flags[idx_to_flip] = 1.0
            
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

