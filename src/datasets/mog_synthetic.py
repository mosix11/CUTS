import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from .base_classification_dataset import BaseClassificationDataset
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
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.cluster_means_: torch.Tensor = None
        self.cluster_generators_: Union[torch.Tensor, List[torch.Tensor]] = None
        self.cluster_class_labels_: torch.Tensor = None # Store mapping: cluster_idx -> class_label

        # Generate the data internally
        self._generate_data()
        
    def set_transformations(self, transfromations):
        self.transfromations = transfromations

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
        self.data = torch.cat(X_list, dim=0)
        self.targets = torch.cat(y_list, dim=0)

        # Shuffle the generated samples and labels together
        perm = torch.randperm(self.n_samples, generator=self.generator)
        self.data = self.data[perm]
        self.targets = self.targets[perm]

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

    def __getitem__(self, idx:int):
        """Returns the feature and label for a given index."""
        if hasattr(self, 'transfromations'):
            return self.transfromations(self.data[idx]), self.targets[idx]

        return self.data[idx], self.targets[idx]

class MoGSynthetic(BaseClassificationDataset):
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        num_samples: int = 100000,
        num_features: int = 512,
        num_classes: int = 30,
        clusters_per_class: Union[str, int, List[int], Tuple[int]] = 'random',
        base_cluster_std: Union[str, float, List[float], Tuple[float]] = 'random',
        covariance_type: Literal['isotropic', 'diagonal', 'full'] = 'full',
        class_sep: float = 1.0,
        intra_class_spread: float = 2.0,
        testset_ratio: float = 0.3,
        seed: int = None,
        **kwargs
    ):
        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / 'MoGSynthetic'
        dataset_dir.mkdir(exist_ok=True, parents=True)

        
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.clusters_per_class = clusters_per_class
        self.base_cluster_std = base_cluster_std
        self.covariance_type = covariance_type
        self.class_sep = class_sep
        self.intra_class_spread = intra_class_spread
        self.testset_ratio = testset_ratio
        
        
        # self.generator = None
        # if seed:
        #     self.seed = seed
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        #     self.generator = torch.Generator()
        #     self.generator.manual_seed(self.seed)
        
        
        super().__init__(
            dataset_name='MoGSynthetic',
            dataset_dir=dataset_dir,
            num_classes=num_classes,
            **kwargs,
            seed=seed
        )
        
        
    def load_train_set(self):
        self._load_data()
        return self.trainset
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        if hasattr(self, 'testset'):
            return self.testset
        else:
            raise RuntimeError('In order to be able to load the test set you have to first call the `load_train_set`.')

        

    def get_identifier(self):
        identifier = 'mog|'
        identifier += f'smpls{self.num_samples}|'
        identifier += f'ftrs{self.num_features}|'
        identifier += f'cls{self.num_classes}'
        return identifier
        
        
    def _load_data(self):
        param_dict = {
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'clusters_per_class': self.clusters_per_class if isinstance(self.clusters_per_class, int) else tuple(self.clusters_per_class) ,
            'base_cluster_std': self.base_cluster_std if isinstance(self.base_cluster_std, float) else tuple(self.base_cluster_std),
            'covariance_type': self.covariance_type,
            'class_sep': self.class_sep,
            'intra_class_spread': self.intra_class_spread,
            'seed': self.seed
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
            if isinstance(self.clusters_per_class, str) and self.clusters_per_class == 'random':
                if not (isinstance(self.base_cluster_std, str) and self.base_cluster_std == 'random'):
                    raise ValueError('If the number of clusters is set to be randomly generated, the stds of the clusters must also be generated randomly.')
                else:
                    self.clusters_per_class = torch.randint(low=1, high=5, size=(self.num_classes,), generator=self.generator).numpy().tolist()
                    self.num_clusters = np.sum(self.clusters_per_class)
                    self.base_cluster_std = torch.rand(self.num_clusters) * (6.0 - 1.0) + 1.0
                    self.base_cluster_std = self.base_cluster_std.numpy().tolist()

            
            self.full_dataset = MoGSyntheticDataset(
                n_samples=self.num_samples,
                n_features=self.num_features,
                n_classes=self.num_classes,
                n_clusters_per_class=self.clusters_per_class,
                base_cluster_std=self.base_cluster_std,
                covariance_type=self.covariance_type,
                class_sep=self.class_sep,
                intra_class_spread=self.intra_class_spread,
                random_state=self.generator,
            )
            torch.save(self.full_dataset, self.dataset_path)
            print(f"Saved dataset to {self.dataset_path}")
    

        trainset, testset = random_split(
            self.full_dataset,
            [1 - self.testset_ratio, self.testset_ratio],
            generator=self.generator,
        )
        
        train_features = torch.stack([x for x, _ in trainset])
        feature_mean = train_features.mean(dim=0)
        feature_std = train_features.std(dim=0) + 1e-8 
        
        normalization_trnsfm = lambda x: (x - feature_mean) / feature_std
        
        trainset.dataset.set_transformations(normalization_trnsfm)
        testset.dataset.set_transformations(normalization_trnsfm)
        
        self.trainset = trainset
        self.testset = testset
        
        

