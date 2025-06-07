import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from .utils import LabelRemapper, NoisyDataset, apply_label_noise

import os
import sys
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List

class CIFAR10:
    

    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 256,
        img_size: tuple = (32, 32),
        subsample_size: Tuple[int, int] = (-1, -1), # (TrainSet size, TestSet size)
        class_subset: list = [],
        remap_labels: bool = False,
        balance_classes: bool = False,
        label_noise: float = 0.0,
        grayscale: bool = False,
        augmentations: list = [],
        normalize_imgs: bool = False,
        flatten: bool = False,  # Whether to flatten images to vectors
        valset_ratio: float = 0.05,
        num_workers: int = 2,
        seed: int = None,
    ) -> None:

        super().__init__()

        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / Path("CIFAR10")
        dataset_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_dir = dataset_dir

        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.subsample_size = subsample_size
        self.class_subset = class_subset
        self.remap_labels = remap_labels
        self.balance_classes = balance_classes
        self.label_noise = label_noise
        self.grayscale = grayscale
        self.augmentations = augmentations
        self.normalize_imgs = normalize_imgs
        
        self.flatten = flatten  # Store the flatten argument
        self.trainset_ration = 1 - valset_ratio
        self.valset_ratio = valset_ratio
        
        self.generator = None
        if seed:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        self._init_loaders()
        
    def get_transforms(self, train=True):
        trnsfrms = []

        if self.img_size != (32, 32):
            trnsfrms.append(transforms.Resize(self.img_size))

        if self.grayscale:
            trnsfrms.append(transforms.Grayscale(num_output_channels=1))

        if len(self.augmentations) > 0 and train:
            print('Augmentation active')
            # trnsfrms.append(transforms.RandomCrop(32, padding=4))
            # trnsfrms.append(transforms.RandomHorizontalFlip())    
            trnsfrms.extend(self.augmentations) 

        trnsfrms.extend([
            transforms.ToImage(),  # Convert PIL Image/NumPy to tensor
            transforms.ToDtype(
                torch.float32, scale=True
            ),  # Scale to [0.0, 1.0] and set dtype
        ])

        if self.normalize_imgs:
            mean, std = (
                (0.5,), (0.5,)
                if self.grayscale
                else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Values Specific to CIFAR
            )
            trnsfrms.append(transforms.Normalize(mean, std))

        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))  # Use Lambda for flattening

        return transforms.Compose(trnsfrms)

    def get_train_dataloader(self):
        return self.train_loader

    def get_val_dataloader(self):
        return self.val_loader

    def get_test_dataloader(self):
        return self.test_loader
    
    def get_identifier(self):
        identifier = 'cifar10|'
        identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier



    def _get_balanced_subset(self, dataset: Dataset, total_size: int, class_subset: list, generator: torch.Generator) -> Subset:
        """
        Performs stratified sampling to get a balanced subset of data.
        """
        num_classes = len(class_subset)
        if total_size == -1 or total_size is None:
             # If subsample size for this split is not specified, do nothing.
             # This can happen if subsample_size is e.g. (1000, -1)
             return dataset

        if total_size % num_classes != 0:
            raise ValueError(
                f"For balanced sampling, the subsample size ({total_size}) must be "
                f"perfectly divisible by the number of classes ({num_classes})."
            )
        
        samples_per_class = total_size // num_classes
        print(f"Performing balanced sampling: {samples_per_class} samples per class for {num_classes} classes.")

        targets = torch.tensor(dataset.targets)
        
        final_indices = []
        for class_label in class_subset:
            # Find all indices in the original dataset for the current class
            class_indices = torch.where(targets == class_label)[0]
            
            if len(class_indices) < samples_per_class:
                raise ValueError(
                    f"Cannot sample {samples_per_class} instances for class {class_label}, "
                    f"as only {len(class_indices)} are available in the dataset."
                )
                
            # Randomly select 'samples_per_class' indices
            perm = torch.randperm(len(class_indices), generator=generator)
            selected_indices = class_indices[perm[:samples_per_class]]
            final_indices.extend(selected_indices.tolist())
            
        # Shuffle the final list to ensure batches are not ordered by class
        final_indices_tensor = torch.tensor(final_indices)
        shuffled_perm = torch.randperm(len(final_indices_tensor), generator=generator)
        shuffled_final_indices = final_indices_tensor[shuffled_perm].tolist()

        return Subset(dataset, shuffled_final_indices)

    def _init_loaders(self):
        train_dataset = datasets.CIFAR10(
            root=self.dataset_dir, train=True, transform=self.get_transforms(train=True), download=True
        )
        test_dataset = datasets.CIFAR10(
            root=self.dataset_dir, train=False, transform=self.get_transforms(train=False), download=True
        )
        
        
        use_balanced_sampling = self.balance_classes and self.class_subset and self.subsample_size != (-1, -1)

        if use_balanced_sampling:
            # The new balanced (stratified) subsampling path
            train_dataset = self._get_balanced_subset(train_dataset, self.subsample_size[0], self.class_subset, self.generator)
            test_dataset = self._get_balanced_subset(test_dataset, self.subsample_size[1], self.class_subset, self.generator)
        else:
            # The original path for unbalanced or no subsampling
            if self.class_subset:
                train_idxs = [i for i, lbl in enumerate(train_dataset.targets) if lbl in self.class_subset]
                train_dataset = Subset(train_dataset, train_idxs)

                test_idxs = [i for i, lbl in enumerate(test_dataset.targets) if lbl in self.class_subset]
                test_dataset = Subset(test_dataset, test_idxs)
            
            if self.subsample_size != (-1, -1):
                if self.subsample_size[0] != -1:
                    train_indices = torch.randperm(len(train_dataset), generator=self.generator)[:self.subsample_size[0]]
                    train_dataset = Subset(train_dataset, train_indices.tolist())
                if self.subsample_size[1] != -1:
                    test_indices = torch.randperm(len(test_dataset), generator=self.generator)[:self.subsample_size[1]]
                    test_dataset = Subset(test_dataset, test_indices.tolist())


        if self.valset_ratio == 0.0:
            trainset = train_dataset
            valset = None
        else:
            trainset, valset = random_split(
                train_dataset, [self.trainset_ration, self.valset_ratio], generator=self.generator
            )
        testset = test_dataset
        
        if self.class_subset and self.remap_labels:
            mapping = {orig: new for new, orig in enumerate(self.class_subset)}
            trainset = LabelRemapper(trainset, mapping)
            if valset is not None:
                valset = LabelRemapper(valset, mapping)
            testset  = LabelRemapper(testset,  mapping)
   
        if self.label_noise > 0.0:
            trainset = apply_label_noise(trainset, self.label_noise, self.class_subset, self.generator)
            
        trainset = NoisyDataset(trainset, is_noisy_applied=self.label_noise > 0.0)
        if valset is not None:
            valset = NoisyDataset(valset, is_noisy_applied=False)
        testset = NoisyDataset(testset, is_noisy_applied=False)
        
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = self._build_dataloader(valset) if valset else None
        self.test_loader = self._build_dataloader(testset)

    def _build_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False, # TODO fix
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator
        )