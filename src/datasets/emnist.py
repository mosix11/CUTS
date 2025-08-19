import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from .base_classification_dataset import BaseClassificationDataset
from .dataset_wrappers import DatasetWithIndex, LabelRemapper, NoisyClassificationDataset, BinarizedClassificationDataset

import os
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List, Union, Dict


class EMNIST(BaseClassificationDataset):
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        img_size: Union[tuple, list] = (28, 28),
        grayscale: bool = True,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        train_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        val_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        **kwargs
    ) -> None:
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten
        self.augmentations = [] if augmentations == None else augmentations
        
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
        if (train_transforms or val_transforms) and (augmentations != None):
            raise ValueError('You should either pass augmentations, or train and validation transforms.')
        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / 'EMNIST'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        super().__init__(
            dataset_name='EMNIST',
            dataset_dir=dataset_dir,
            num_classes=26,
            **kwargs,  
        )

    # TODO: change the class so it can accept other splits as well
    def load_train_set(self):
        return datasets.EMNIST(root=self.dataset_dir, split="letters", train=True, transform=self.get_transforms(train=True), download=True)
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        return datasets.EMNIST(root=self.dataset_dir, split="letters", train=False, transform=self.get_transforms(train=False), download=True)


    def get_transforms(self, train=True):
        if self.train_transforms and train:
            return self.train_transforms
        elif self.val_transforms and not train:
            return self.val_transforms
        
        trnsfrms = []
        if self.img_size != (28, 28):
            trnsfrms.append(transforms.Resize(self.img_size))
        if not self.grayscale:
            trnsfrms.append(transforms.Grayscale(num_output_channels=3))
        if len(self.augmentations) > 0 and train:
            trnsfrms.extend(self.augmentations)
        trnsfrms.extend([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        if self.normalize_imgs:
            trnsfrms.append(transforms.Normalize((0.1307,), (0.3081,))) # Values Specific to MNIST

        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)
        

    def get_class_names(self):
        return ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
                
    def get_identifier(self):
        identifier = 'emnist|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
 