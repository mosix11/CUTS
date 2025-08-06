import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from .base_classification_dataset import BaseClassificationDataset
from .utils import DatasetWithIndex, LabelRemapper, NoisyClassificationDataset, BinarizedClassificationDataset

import os
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List, Union, Dict
import openxlab
from openxlab.dataset import get as openxlab_get
from openxlab.dataset import download as openxlab_download
import dotenv


class Clothing1M(BaseClassificationDataset):
    def __init__(
        self,
        dotenv_path: Path = Path("./.env"),
        img_size: Union[tuple, list] = (32, 32),
        grayscale: bool = False,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        **kwargs
    ) -> None:
        
        if dotenv_path.exists():
            dotenv.load_dotenv('.env')
            
        OPENXLAB_AK = os.getenv("OPENXLAB_AK")
        OPENXLAB_SK = os.getenv("OPENXLAB_SK")
        
        openxlab.login(ak=OPENXLAB_AK, sk=OPENXLAB_SK) 
        openxlab_get(dataset_repo='OpenDataLab/Clothing1M', target_path='./data/Clothing1M') 
        openxlab_download(dataset_repo='OpenDataLab/Clothing1M',source_path='./data/Clothing1M/README.md', target_path='/data/Clothing1M')
        
        # self.img_size = img_size
        # self.grayscale = grayscale
        # self.normalize_imgs = normalize_imgs
        # self.flatten = flatten
        # self.augmentations = [] if augmentations == None else augmentations
        
        # super().__init__(
        #     dataset_name='Clothing1M',
        #     num_classes=10,
        #     **kwargs,  
        # )


    def load_train_set(self):
        # return datasets.CIFAR10(root=self.dataset_dir, train=True, transform=self.get_transforms(train=True), download=True)
        pass
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        # return datasets.CIFAR10(root=self.dataset_dir, train=False, transform=self.get_transforms(train=False), download=True)
        pass

    def get_transforms(self, train=True):
        trnsfrms = []
        if self.img_size != (32, 32):
            trnsfrms.append(transforms.Resize(self.img_size))
        if self.grayscale:
            trnsfrms.append(transforms.Grayscale(num_output_channels=1))
        if len(self.augmentations) > 0 and train:
            trnsfrms.extend(self.augmentations)
        trnsfrms.extend([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        if self.normalize_imgs:
            mean, std = ((0.5,), (0.5,)) if self.grayscale else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)


    def get_identifier(self):
        identifier = 'cifar10|'
        # identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    