import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from .base_classification_dataset import BaseClassificationDataset
from typing import Tuple, List, Union, Dict
from pathlib import Path
from torch.utils.data import Subset

import torch.distributed as dist

class Food101(BaseClassificationDataset):
    DEFAULT_SEED = 1111
    
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        img_size: Union[tuple, list] = (224, 224),
        grayscale: bool = False,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        train_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        val_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        val_from_test_size: int = None,
        val_split_seed: int = DEFAULT_SEED,
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
        dataset_dir = data_dir / 'Food101'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        
        self.val_from_test_size = 0 if val_from_test_size == None else val_from_test_size
        self.val_split_seed = val_split_seed
        
        super().__init__(
            dataset_name='Food101',
            dataset_dir=dataset_dir,
            num_classes=101,
            **kwargs,  
        )


    def load_train_set(self):
        self.train_transforms = self.get_transforms(train=True)
        root = self.dataset_dir

        if self.is_distributed():
            if self.is_node_leader():
                _ = datasets.Food101(root=root, split="train", download=True)  # pre-download only
            dist.barrier()
            trainset = datasets.Food101(root=root, split="train",
                                        transform=self.train_transforms, download=False)
        else:
            trainset = datasets.Food101(root=root, split="train",
                                        transform=self.train_transforms, download=True)

        self._class_names = trainset.classes
        return trainset

    def load_validation_set(self):
        if self.val_from_test_size != None:
            self.train_transforms = self.get_transforms(train=True)
            root = self.dataset_dir
            if self.is_distributed():
                if self.is_node_leader():
                    _ = datasets.Food101(root=root, split="test", download=True)  # pre-download only
                dist.barrier()
                testset = datasets.Food101(root=root, split="test",
                                        transform=self.train_transforms, download=False)
            else:
                testset = datasets.Food101(root=root, split="test",
                                        transform=self.train_transforms, download=True)
            
            valset, _ = self._random_subset(testset, size=self.val_from_test_size, seed=self.val_split_seed)
            return valset
        return None

    def load_test_set(self):
        self.val_transforms = self.get_transforms(train=False)
        root = self.dataset_dir

        if self.is_distributed():
            if self.is_node_leader():
                _ = datasets.Food101(root=root, split="test", download=True)  # pre-download only
            dist.barrier()
            testset = datasets.Food101(root=root, split="test",
                                    transform=self.val_transforms, download=False)
        else:
            testset = datasets.Food101(root=root, split="test",
                                    transform=self.val_transforms, download=True)

        if self.val_from_test_size != None:
            _, testset = self._random_subset(testset, size=self.val_from_test_size, seed=self.val_split_seed)
        return testset
    
    def _random_subset(self, ds, size: int, seed: int):
        n = len(ds)
        k = max(0, min(size, n))

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)

        idx_selected = perm[:k].tolist()
        idx_complement = perm[k:].tolist()

        return Subset(ds, idx_selected), Subset(ds, idx_complement)

    def get_transforms(self, train=True):
        if self._train_transforms and train:
            return self._train_transforms
        elif self._val_transforms and not train:
            return self._val_transforms
        
        trnsfrms = []
        if self.img_size != (224, 224):
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
            mean, std = ((0.5,), (0.5,)) if self.grayscale else ((0.5459, 0.4433, 0.3439), (0.2680, 0.2658, 0.2729))
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)


    def get_class_names(self):
        return self._class_names 

    def get_identifier(self):
        identifier = 'food101|'
        # identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
    

    



            