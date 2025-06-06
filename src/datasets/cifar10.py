import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from .utils import LabelRemapper

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


    def _apply_label_noise(self, dataset):
        num_samples = len(dataset)
        num_classes = len(self.class_subset) if self.class_subset else 10

        # Generate random numbers to decide which labels to flip
        noise_mask = torch.rand(num_samples, generator=self.generator) < self.label_noise

        # Get the original labels
        if isinstance(dataset, Subset):
            original_labels = dataset.dataset.targets[dataset.indices].clone().detach()
        else:
            original_labels = dataset.targets.clone().detach()

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
        if isinstance(dataset, Subset):
            dataset.dataset.targets = noisy_labels.tolist()
        else:
            dataset.targets = noisy_labels.tolist()
        return dataset



    def _init_loaders(self):
        train_dataset = datasets.CIFAR10(
            root=self.dataset_dir,
            train=True,
            transform=self.get_transforms(train=True),
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            root=self.dataset_dir,
            train=False,
            transform=self.get_transforms(train=False),
            download=True,
        )
        
        # Filter samples based on the provided classes
        if self.class_subset != None and len(self.class_subset) >= 1:
            train_idxs = [
                i for i, lbl in enumerate(train_dataset.targets)
                if lbl in self.class_subset
            ]
            train_dataset = Subset(train_dataset, train_idxs)

            test_idxs = [
                i for i, lbl in enumerate(test_dataset.targets)
                if lbl in self.class_subset
            ]
            test_dataset = Subset(test_dataset, test_idxs)
        
        # Subsample the dataset uniformly
        if self.subsample_size != (-1, -1):
            train_indices = torch.randperm(len(train_dataset), generator=self.generator)[:self.subsample_size[0]]
            test_indices = torch.randperm(len(train_dataset), generator=self.generator)[:self.subsample_size[1]]
            train_dataset = Subset(train_dataset, train_indices.tolist())
            test_dataset = Subset(test_dataset, test_indices.tolist())

        if self.valset_ratio == 0.0:
            trainset = train_dataset
            valset = None
        else:
            
            trainset, valset = random_split(
                train_dataset,
                [self.trainset_ration, self.valset_ratio],
                generator=self.generator,
            )
        testset = test_dataset
            
        if self.label_noise > 0.0:
            trainset = self._apply_label_noise(trainset)
            
        if self.class_subset != None and len(self.class_subset) >= 1:
            
            mapping = {orig: new for new, orig in enumerate(self.class_subset)}
            trainset = LabelRemapper(trainset, mapping)
            if valset is not None:
                valset = LabelRemapper(valset, mapping)
            testset  = LabelRemapper(testset,  mapping)
   
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = (
            self._build_dataloader(valset) if self.valset_ratio > 0 else None
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