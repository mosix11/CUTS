import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from .base_classification_dataset import BaseClassificationDataset
from typing import Tuple, List, Union, Dict
from pathlib import Path

class SVHN(BaseClassificationDataset):
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        img_size: Union[tuple, list] = (32, 32),
        grayscale: bool = False,
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
        
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        
        if (train_transforms or val_transforms) and (augmentations != None):
            raise ValueError('You should either pass augmentations, or train and validation transforms.')
        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / 'SVHN'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        super().__init__(
            dataset_name='SVHN',
            dataset_dir=dataset_dir,
            num_classes=10,
            **kwargs,  
        )


    def load_train_set(self):
        self.train_transforms = self.get_transforms(train=True)
        return datasets.SVHN(root=self.dataset_dir, split="train", transform=self.train_transforms, download=True)
    
    def load_validation_set(self):
        # 531,131 samples in "extra" split which are less curated.
        # return datasets.SVHN(root=self.dataset_dir, split="extra", transform=self.get_transforms(train=True), download=True)
        return None
    
    def load_test_set(self):
        self.val_transforms = self.get_transforms(train=False)
        return datasets.SVHN(root=self.dataset_dir, split="test", transform=self.val_transforms, download=True)

    def get_transforms(self, train=True):
        if self._train_transforms and train:
            return self._train_transforms
        elif self._val_transforms and not train:
            return self._val_transforms
        
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
            mean, std = ((0.5,), (0.5,)) if self.grayscale else ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)


    def get_class_names(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def get_identifier(self):
        identifier = 'svhn|'
        # identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
    

    



            