import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from pathlib import Path
from typing import List, Tuple, Union, Literal, Optional, Sequence
import random
import numpy as np
import json
import hashlib


class MoGSyntheticDataset(Dataset):
    """
    A PyTorch Dataset class to generate synthetic classification data.

    Generates data based on multimodal Gaussian mixtures, similar to
    sklearn's make_classification, upon initialization. Provides access
    to the generated features and labels. Use with torch.utils.data.random_split
    to create training, validation, and test sets.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        n_clusters_per_class: Union[int, List[int], Tuple[int]] = 1,
        base_cluster_std: Union[float, List[float], Tuple[float]] = 1.0,
        covariance_type: Literal['isotropic', 'diagonal', 'full'] = 'isotropic',
        class_sep: float = 1.0,
        intra_class_spread: float = 0.5,
        # label_noise: float = 0.0,
        random_state: Optional[Union[int, torch.Generator]] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            n_samples (int): Total number of samples.
            n_features (int): Number of features per sample.
            n_classes (int): Number of classes.
            n_clusters_per_class (int or list/tuple): Number of Gaussian clusters per class.
                If int, same number for all classes. If list/tuple, specifies count
                for each class, length must equal n_classes.
            base_cluster_std (float or list/tuple): Base standard deviation(s) for clusters.
                If float, used for all clusters. If list/tuple, length must match
                n_classes * n_clusters_per_class. Acts as a scaling factor.
            covariance_type (str): Type of covariance matrix for clusters:
                'isotropic': Scaled identity matrix (sigma^2 * I).
                'diagonal': Diagonal matrix with random positive entries, scaled by base_cluster_std.
                'full': Random positive semi-definite matrix, scaled by base_cluster_std.
            intra_class_spread (float): Factor scaling spread of cluster means
                                       within the same class around the class centroid.
            label_noise (float): Fraction of labels to randomly flip (0.0 to 1.0).
            random_state (int or torch.Generator): Seed or generator for reproducibility.
            dtype (torch.dtype): Data type for the feature tensor X.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_sep = class_sep
        self.intra_class_spread = intra_class_spread 
        self.cluster_std_param = base_cluster_std
        self.covariance_type = covariance_type
        # self.label_noise = label_noise
        self.dtype = dtype

        # --- Input Validation ---
        if not isinstance(n_classes, int) or n_classes < 1:
             raise ValueError("n_classes must be a positive integer.")
        if not isinstance(n_features, int) or n_features < 1:
             raise ValueError("n_features must be a positive integer.")

        # Validate and normalize n_clusters_per_class
        if isinstance(n_clusters_per_class, int):
            if n_clusters_per_class < 1:
                raise ValueError("If n_clusters_per_class is int, it must be positive.")
            self.n_clusters_per_class_list: List[int] = [n_clusters_per_class] * n_classes
        elif isinstance(n_clusters_per_class, (list, tuple)):
            if len(n_clusters_per_class) != n_classes:
                raise ValueError("If n_clusters_per_class is a list/tuple, its length must equal n_classes.")
            if not all(isinstance(n, int) and n >= 1 for n in n_clusters_per_class):
                 raise ValueError("Each element in n_clusters_per_class list/tuple must be a positive integer.")
            self.n_clusters_per_class_list: List[int] = list(n_clusters_per_class)
        else:
            raise TypeError("n_clusters_per_class must be an int, list, or tuple.")

        self.num_total_clusters: int = sum(self.n_clusters_per_class_list)
        if self.num_total_clusters == 0:
             raise ValueError("Total number of clusters cannot be zero.") # Should be caught by checks above

        if self.covariance_type not in ['isotropic', 'diagonal', 'full']:
            raise ValueError("covariance_type must be 'isotropic', 'diagonal', or 'full'")

        # Validate cluster_std length if it's a list/tuple
        if isinstance(self.cluster_std_param, (list, tuple)):
             if len(self.cluster_std_param) != self.num_total_clusters:
                 raise ValueError(f"Length of cluster_std ({len(self.cluster_std_param)}) must match "
                                  f"the total number of clusters ({self.num_total_clusters})")

        # Initialize generator
        if random_state is None:
            self.generator = torch.Generator()
        elif isinstance(random_state, int):
            self.generator = torch.Generator().manual_seed(random_state)
        elif isinstance(random_state, torch.Generator):
            self.generator = random_state
        else:
            raise ValueError("random_state must be None, int, or torch.Generator")

        # --- Attributes to be populated by _generate_data ---
        self.features: torch.Tensor = None
        self.labels: torch.Tensor = None
        self.cluster_means_: torch.Tensor = None
        self.cluster_generators_: Union[torch.Tensor, List[torch.Tensor]] = None
        self.cluster_class_labels_: torch.Tensor = None # Store mapping: cluster_idx -> class_label

        # Generate the data internally
        self._generate_data()

    def _generate_data(self):
        """Internal method to generate features (X) and labels (y)."""

        # Ensure n_samples is sufficient
        if self.n_samples < self.num_total_clusters:
             raise ValueError(f"n_samples ({self.n_samples}) must be at least the total number of clusters ({self.num_total_clusters}).")

        # Distribute samples as evenly as possible among clusters
        base_samples_per_cluster = self.n_samples // self.num_total_clusters
        remainder = self.n_samples % self.num_total_clusters
        samples_per_cluster_list = [base_samples_per_cluster + (1 if i < remainder else 0) for i in range(self.num_total_clusters)]
        assert sum(samples_per_cluster_list) == self.n_samples # Verify exact sample count

        X_list = []
        y_list = []
        generators_list = [] # To store std scalars, std vectors, or Cholesky factors

        # --- Generate Cluster Means ---
        # 1. Generate base centroids for each class, separated by class_sep
        class_centroids = torch.randn(
            self.n_classes, self.n_features,
            generator=self.generator, dtype=self.dtype
        ) * (self.class_sep * self.n_features**0.5) # Adjusted scaling heuristic

        # 2. Generate specific cluster means by adding offset based on intra_class_spread
        cluster_means = torch.empty(
            self.num_total_clusters, self.n_features, dtype=self.dtype
        )
        # Create mapping from global cluster index to class index
        _cluster_class_map_list = []
        current_cluster_idx = 0
        for class_idx, num_clusters in enumerate(self.n_clusters_per_class_list):
             _cluster_class_map_list.extend([class_idx] * num_clusters)
             for _ in range(num_clusters):
                 # Get the centroid for the cluster's assigned class
                 centroid = class_centroids[class_idx]
                 # Add random offset scaled by intra_class_spread
                 # Only add offset if there's more than one cluster for the class OR if spread > 0
                 if num_clusters > 1 or self.intra_class_spread > 1e-6:
                     mean_offset = torch.randn(
                         self.n_features, generator=self.generator, dtype=self.dtype
                     ) * self.intra_class_spread
                 else:
                     mean_offset = 0.0 # No spread if only one cluster and spread is zero

                 cluster_means[current_cluster_idx] = centroid + mean_offset
                 current_cluster_idx += 1

        self.cluster_class_labels_ = torch.tensor(_cluster_class_map_list, dtype=torch.int64)


        # --- Prepare Base Cluster Standard Deviations (Scaling Factors) ---
        if isinstance(self.cluster_std_param, (float, int)):
            cluster_base_stds = torch.full((self.num_total_clusters,), float(self.cluster_std_param), dtype=self.dtype)
        else: # List or tuple (already validated length)
            cluster_base_stds = torch.tensor(self.cluster_std_param, dtype=self.dtype)

        # --- Sample points from each cluster ---
        for i in range(self.num_total_clusters):
            mean = cluster_means[i]
            base_std = cluster_base_stds[i] # Base scaling factor for this cluster
            n_samples_cluster = samples_per_cluster_list[i]

            # Generate standard normal samples first
            Z = torch.randn(
                n_samples_cluster, self.n_features,
                generator=self.generator, dtype=self.dtype
            )

            # Apply transformation based on covariance type
            if self.covariance_type == 'isotropic':
                std_val = base_std
                cluster_X = mean + std_val * Z
                generators_list.append(std_val.clone().detach())

            elif self.covariance_type == 'diagonal':
                # Use positive random values (e.g., gamma or scaled rand) for stability
                # rand is [0, 1), so 0.5 + rand is [0.5, 1.5) -> avg 1.0
                std_vector = base_std * (0.5 + torch.rand(self.n_features, generator=self.generator, dtype=self.dtype))
                cluster_X = mean + std_vector * Z
                generators_list.append(std_vector.clone().detach())

            elif self.covariance_type == 'full':
                A = torch.randn(self.n_features, self.n_features, generator=self.generator, dtype=self.dtype)
                Sigma_shape = A @ A.T + torch.eye(self.n_features, dtype=self.dtype) * 1e-5
                try:
                    L = torch.linalg.cholesky(Sigma_shape)
                except torch.linalg.LinAlgError:
                     print(f"Warning: Cholesky decomposition failed for cluster {i}. Using diagonal approximation.")
                     L = torch.diag(torch.sqrt(torch.diag(Sigma_shape).clamp(min=1e-5)))

                scaled_L = base_std * L
                cluster_X = mean + (scaled_L @ Z.T).T
                generators_list.append(scaled_L.clone().detach())

            X_list.append(cluster_X)

            # Assign class label based on the pre-computed map
            class_label = self.cluster_class_labels_[i]
            cluster_y = torch.full((n_samples_cluster,), class_label.item(), dtype=torch.int64)
            y_list.append(cluster_y)

        # Concatenate all generated data
        self.features = torch.cat(X_list, dim=0)
        self.labels = torch.cat(y_list, dim=0)

        # Shuffle the generated samples and labels together
        perm = torch.randperm(self.n_samples, generator=self.generator)
        self.features = self.features[perm]
        self.labels = self.labels[perm]

        # Store the generated cluster parameters (means already stored)
        self.cluster_means_ = cluster_means # Shape: (num_total_clusters, n_features)
        if self.covariance_type == 'isotropic' or self.covariance_type == 'diagonal':
             # Stack scalar stds or std vectors
             self.cluster_generators_ = torch.stack(generators_list, dim=0)
        else: # 'full', store list of Cholesky factors
             self.cluster_generators_ = generators_list # List of Tensors

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Returns the feature and label for a given index."""
        # (Code for __getitem__ remains the same)
        if not isinstance(idx, int):
             raise TypeError("Index must be an integer. Slicing is not directly supported, use DataLoader or random_split.")
        if not (0 <= idx < self.n_samples):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {self.n_samples}")
        return self.features[idx], self.labels[idx]

class MoGSynthetic:
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 256,
        num_samples: int = 100000,
        num_features: int = 512,
        num_classes: int = 10,
        clusters_per_class: Union[str, int, List[int], Tuple[int]] = 'random',
        base_cluster_std: Union[str, float, List[float], Tuple[float]] = 'random',
        covariance_type: Literal['isotropic', 'diagonal', 'full'] = 'isotropic',
        class_sep: float = 1.0,
        intra_class_spread: float = 0.5,
        label_noise: float = 0.0,
        train_val_test_ratio: Tuple[float] = (0.7, 0.0, 0.3),
        num_workers: int = 2,
        seed: int = None,
    ):
        super().__init__()
        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir.joinpath(Path("MoGSynthetic"))
        dataset_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_dir = dataset_dir
        
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tot_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.label_noise = label_noise
        
        self.generator = None
        if seed:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)
            
        if np.sum(train_val_test_ratio) != 1.0:
            raise ValueError('The sum of the values passed as `train_val_test_ratio` should be 1!')
        self.train_val_test_ratio = train_val_test_ratio
        
            
        
        param_dict = {
            'num_samples': num_samples,
            'num_features': num_features,
            'num_classes': num_classes,
            'clusters_per_class': clusters_per_class if isinstance(clusters_per_class, int) else tuple(clusters_per_class) ,
            'base_cluster_std': base_cluster_std if isinstance(base_cluster_std, float) else tuple(base_cluster_std),
            'covariance_type': covariance_type,
            'class_sep': class_sep,
            'intra_class_spread': intra_class_spread,
            'seed': seed
        }
        
        # Generate unique identifier
        param_str = json.dumps(param_dict, sort_keys=True)
        self.identifier = hashlib.md5(param_str.encode()).hexdigest()
        self.dataset_path = self.dataset_dir / f"{self.identifier}.pt"
        
        if self.dataset_path.exists():
            print(f"Loading existing dataset from {self.dataset_path}")
            self.full_dataset = torch.load(self.dataset_path, weights_only=False)
        else:
            print(f"Generating new dataset (ID: {self.identifier})")
            if isinstance(clusters_per_class, str) and clusters_per_class == 'random':
                if not (isinstance(base_cluster_std, str) and base_cluster_std == 'random'):
                    raise ValueError('If the number of clusters is set to be randomly generated, the stds of the clusters must also be generated randomly.')
                else:
                    clusters_per_class = torch.randint(low=1, high=5, size=(num_classes,), generator=self.generator).numpy().tolist()
                    num_clusters = np.sum(clusters_per_class)
                    base_cluster_std = torch.rand(num_clusters) * (6.0 - 1.0) + 1.0
                    base_cluster_std = base_cluster_std.numpy().tolist()

            
            self.full_dataset = MoGSyntheticDataset(
                n_samples=num_samples,
                n_features=num_features,
                n_classes=num_classes,
                n_clusters_per_class=clusters_per_class,
                base_cluster_std=base_cluster_std,
                covariance_type=covariance_type,
                class_sep=class_sep,
                intra_class_spread=intra_class_spread,
                random_state=self.generator,
            )
            torch.save(self.full_dataset, self.dataset_path)
            print(f"Saved dataset to {self.dataset_path}")
        
        
        
        self._init_loaders()
        
    def get_train_dataloader(self):
        return self.train_loader

    def get_val_dataloader(self):
        return self.val_loader

    def get_test_dataloader(self):
        return self.test_loader
    
    def get_identifier(self):
        identifier = 'mog|'
        identifier += f'smpls{self.tot_samples}|'
        identifier += f'ftrs{self.num_features}|'
        identifier += f'cls{self.num_classes}'
        identifier += f'ln{self.label_noise}|'
        return identifier
    
    def _apply_label_noise(self, dataset):
        num_samples = len(dataset)
        num_classes = self.num_classes

        # Generate random numbers to decide which labels to flip
        noise_mask = torch.rand(num_samples, generator=self.generator) < self.label_noise


        # Get the original labels
        # original_labels = dataset.labels.clone().detach()
        if isinstance(dataset, Subset):
            original_labels = dataset.dataset.labels[dataset.indices].clone().detach()
        else:
            original_labels = dataset.dataset.labels.clone().detach()
        

        # Generate random incorrect labels
        random_labels = torch.randint(0, num_classes, (num_samples,), generator=self.generator)

        # Ensure the random labels are different from the original labels
        incorrect_mask = (random_labels == original_labels)
        while incorrect_mask.any():
            new_random_labels = torch.randint(0, num_classes, (incorrect_mask.sum(),), generator=self.generator)
            random_labels[incorrect_mask] = new_random_labels
            incorrect_mask = (random_labels == original_labels)

        # Apply the noise to the targets
        noisy_labels = torch.where(noise_mask, random_labels, original_labels)

        # Update the dataset targets
        # dataset.dataset.labels[dataset.indices] = noisy_labels
        # return dataset

        # Update the dataset targets
        if isinstance(dataset, Subset):
            # Assign noisy labels to the original dataset's targets at the Subset indices
            dataset.dataset.labels[dataset.indices] = noisy_labels
            if not hasattr(dataset.dataset, 'is_noisy'):
                dataset.dataset.is_noisy = torch.zeros(len(dataset.dataset.labels), dtype=torch.bool)
            dataset.dataset.is_noisy[dataset.indices] = noise_mask
        else:
            # Directly update the dataset's targets (avoids converting to list)
            dataset.labels = noisy_labels
            dataset.is_noisy = noise_mask

        return dataset
    
    
    def _init_loaders(self):
        
        trainset, valset, testset = random_split(
            self.full_dataset,
            self.train_val_test_ratio,
            generator=self.generator,
        )
        
        if self.label_noise > 0.0:
            trainset = self._apply_label_noise(trainset)
        
        train_features = torch.stack([x for x, _ in trainset])
        feature_mean = train_features.mean(dim=0)
        feature_std = train_features.std(dim=0) + 1e-8 
        
        class NormalizedDataset(Dataset):
            def __init__(self, set, feature_mean, feature_std, is_noisy=False):
                self.set = set
                self.feature_mean = feature_mean
                self.feature_std = feature_std
                self.is_noisy = is_noisy
                
            def __getitem__(self, idx):
                x, y = self.set[idx]
                normalized_x = (x - self.feature_mean) / self.feature_std

                # Get the is_noisy flag
                if self.is_noisy:
                    # For training set, retrieve the specific is_noisy flag for the current sample
                    original_idx = self.set.indices[idx] if isinstance(self.set, Subset) else idx
                    is_noisy_flag = self.set.dataset.is_noisy[original_idx]
                else:
                    # For validation/test sets, or if 'is_noisy' not present, it's always False
                    is_noisy_flag = torch.tensor(False, dtype=torch.bool)

                return normalized_x, y, is_noisy_flag
                
            def __len__(self):
                return len(self.set)
        
        trainset = NormalizedDataset(trainset, feature_mean, feature_std, is_noisy=True if self.label_noise > 0.0 else False)
        valset = NormalizedDataset(valset, feature_mean, feature_std) if valset else None
        testset = NormalizedDataset(testset, feature_mean, feature_std)

        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = (
            self._build_dataloader(valset) if self.train_val_test_ratio[1] > 0 else None
        )
        self.test_loader = self._build_dataloader(testset)

    def _build_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator
        )
        return dataloader
