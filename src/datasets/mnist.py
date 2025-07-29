import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from .base_classification_dataset import BaseClassificationDataset
from .utils import DatasetWithIndex, LabelRemapper, NoisyClassificationDataset, BinarizedClassificationDataset

import os
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List, Union, Dict


class MNIST(BaseClassificationDataset):
    def __init__(
        self,
        img_size: Union[tuple, list] = (28, 28),
        grayscale: bool = True,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        **kwargs
    ) -> None:
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten
        self.augmentations = [] if augmentations == None else augmentations
        
        super().__init__(
            dataset_name='MNIST',
            num_classes=10,
            **kwargs,  
        )

    def load_train_set(self):
        return datasets.MNIST(root=self.dataset_dir, train=True, transform=self.get_transforms(train=True), download=True)
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        return datasets.MNIST(root=self.dataset_dir, train=False, transform=self.get_transforms(train=False), download=True)


    def get_transforms(self, train=True):
        
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
        

    
    def get_identifier(self):
        identifier = 'mnist|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
 