import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset
from .base_classification_dataset import BaseClassificationDataset
from typing import Tuple, List, Union, Dict
from pathlib import Path


class EuroSAT(BaseClassificationDataset):
    DEFAULT_TEST_SIZE = 2700
    DEFAULT_SEED = 1337

    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        img_size: Union[tuple, list] = (64, 64),
        grayscale: bool = False,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        train_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        val_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        test_size: int = DEFAULT_TEST_SIZE,
        seed: int = DEFAULT_SEED,
        **kwargs
    ) -> None:
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten
        self.augmentations = [] if augmentations is None else augmentations

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        if (train_transforms or val_transforms) and (augmentations is not None):
            raise ValueError('You should either pass augmentations, or train and validation transforms.')

        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / 'EuroSAT'
        dataset_dir.mkdir(exist_ok=True, parents=True)

        self.test_size = int(test_size)
        self.seed = int(seed)
        self._split_indices = None  # (train_idx, test_idx), computed lazily

        super().__init__(
            dataset_name='EuroSAT',
            dataset_dir=dataset_dir,
            num_classes=10,
            **kwargs,
        )

    def load_train_set(self):
        train_idx, _ = self._get_split_indices()
        base = self._base_dataset(transform=self.get_transforms(train=True))
        self._class_names = base.classes
        return Subset(base, train_idx)

    def load_validation_set(self):
        return None

    def load_test_set(self):
        _, test_idx = self._get_split_indices()
        base = self._base_dataset(transform=self.get_transforms(train=False))
        return Subset(base, test_idx)

    def get_transforms(self, train=True):
        if self.train_transforms and train:
            return self.train_transforms
        elif self.val_transforms and not train:
            return self.val_transforms

        trnsfrms = []
        if self.img_size != (64, 64):
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
            mean, std = ((0.5,), (0.5,)) if self.grayscale else (
                (0.3443, 0.3809, 0.4082), (0.2038, 0.1364, 0.1199)
            )
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)

    def get_class_names(self):
        return self._class_names 

    def get_identifier(self):
        identifier = 'eurosat|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier


    def _base_dataset(self, transform):
        # EuroSAT in torchvision has no split; it’s a single 27k-sample dataset.
        # Setting download=True in both calls is fine—download happens once.
        return datasets.EuroSAT(root=self.dataset_dir, transform=transform, download=True)

    def _get_split_indices(self):
        if self._split_indices is not None:
            return self._split_indices

        # Build a minimal dataset once to read its length deterministically
        base_for_len = datasets.EuroSAT(root=self.dataset_dir, transform=None, download=True)
        n = len(base_for_len)  # expected 27000
        if not (0 < self.test_size < n):
            raise ValueError(f"Invalid test_size={self.test_size}; must be in (0, {n}).")

        g = torch.Generator()
        g.manual_seed(self.seed)  # deterministic across machines / runs
        perm = torch.randperm(n, generator=g).tolist()

        test_idx = perm[:self.test_size]
        train_idx = perm[self.test_size:]

        # Keep a stable ordering within each subset (optional but nice)
        test_idx.sort()
        train_idx.sort()

        self._split_indices = (train_idx, test_idx)
        return self._split_indices