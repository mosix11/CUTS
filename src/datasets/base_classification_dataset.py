import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from .dataset_wrappers import DatasetWithIndex, LabelRemapper, NoisyClassificationDataset, BinarizedClassificationDataset

import os
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List, Union, Dict
from abc import ABC, abstractmethod

class BaseClassificationDataset(ABC):
    def __init__(
        self,
        dataset_name:str = None,
        dataset_dir:Path = None,
        num_classes:int = None,
        batch_size: int = 256,
        subsample_size: Union[tuple, list] = (-1, -1),
        class_subset: list = [],
        remap_labels: bool = False,
        balance_classes: bool = False,
        heldout_conf: Union[None, float, Dict[int, float]] = None,
        valset_ratio: float = 0.0,
        num_workers: int = 2,
        seed: int = None,
    ) -> None:
        super().__init__()

        if num_classes == None:
            raise ValueError("Number of classes must be specified for initialization.")
        
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subsample_size = subsample_size
        self.class_subset = sorted(class_subset) if class_subset else None
        self.remap_labels = remap_labels
        self.balance_classes = balance_classes
        self.heldout_conf = heldout_conf
        self.valset_ratio = valset_ratio
        self.trainset_ratio = 1 - self.valset_ratio

        if self.class_subset:
            self.available_classes = self.class_subset
        else:
            self.available_classes = list(range(num_classes))

        self.generator = None
        if seed:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.generator = torch.Generator().manual_seed(self.seed)

        self._init_loaders()
        
        
    @abstractmethod
    def load_train_set(self):
        """
        This method must be implemented by all subclasses.
        The method should return a training dataset which is a subclass of torch.data.Dataset.
        """
        
        pass
    
    @abstractmethod
    def load_validation_set(self):
        """
        This method must be implemented by all subclasses.
        The method should return a validation dataset which is a subclass of torch.data.Dataset.
        """
        pass
    
    @abstractmethod
    def load_test_set(self):
        """
        This method must be implemented by all subclasses.
        The method should return a test dataset which is a subclass of torch.data.Dataset.
        """
        pass
    
    @abstractmethod
    def get_class_names(self):
        """
        This method must be implemented by all subclasses.
        The method should return the class names as a list, matching the order of the label indices.
        """
        pass
    
    
    @abstractmethod
    def get_identifier(self):
        """
        This method must be implemented by all subclasses.
        The method should return a string specifying the dataset identifier.
        """
        pass
        
    def get_train_dataloader(self):
        return self.train_loader
    
    def reset_train_dl(self):
        self.train_loader = self._build_dataloader(self.trainset, shuffle=True)

    def get_val_dataloader(self):
        return self.val_loader

    def get_test_dataloader(self):
        return self.test_loader

    def get_heldout_dataloader(self):
        return self.heldout_loader
    
    
    def get_trainset(self):
        return self.trainset
    
    def set_trainset(self, set, shuffle=False):
        self.trainset = set
        self.train_loader = self._build_dataloader(self.trainset, shuffle=shuffle)
    
    def get_valset(self):
        return self.valset
    
    def set_valset(self, set):
        self.valset = set
        self.val_loader = self._build_dataloader(self.valset, shuffle=False)
    
    def get_testset(self):
        return self.testset
    
    def set_testset(self, set):
        self.testset = set
        self.test_loader = self._build_dataloader(self.testset, shuffle=False)
    
    def get_heldoutset(self):
        return self.heldout_set
    
    def set_heldoutset(self, set, shuffle=False):
        self.heldout_set = set
        self.heldout_loader = self._build_dataloader(self.heldout_set, shuffle=shuffle)
        
            
    def get_generator(self):
        return self.generator
    
    def set_generator(self, gen):
        self.generator = gen
        
    def set_generator_seed(self, seed):
        self.generator.manual_seed(seed)
        
    def get_available_classes(self):
        return self.available_classes
    
    def get_num_classes(self):
        return len(self.available_classes)
    
    def subset_set(self, set='Train', indices=[]):
        dataset = self._get_set(set)
        dataset = Subset(dataset, indices)
        self._set_set(set, dataset)
    
    def replace_heldout_as_train_dl(self):
        self.train_loader = self._build_dataloader(self.heldout_set, shuffle=True)
    
    
    def change_batch_size(self, batch_size:int):
        self.batch_size = batch_size
        self.train_loader = self._build_dataloader(self.trainset, shuffle=True)
        self.val_loader = self._build_dataloader(self.valset) if self.valset else None
        self.test_loader = self._build_dataloader(self.testset)
        self.heldout_loader = self._build_dataloader(self.heldout_set) if self.heldout_set else None
        
        
        
    def binarize_set(self, set='Train', target_class=-1):
        if target_class not in self.available_classes:
            raise ValueError('Target class is not in available classes.')
        dataset = self._get_set(set)
        dataset = BinarizedClassificationDataset(dataset, target_class)
        self._set_set(set, dataset)
        
    def inject_noise(self, set='Train', noise_rate=0.0, noise_type='symmetric', T_mat=None, target_class=None, seed=None, generator=None):
        dataset = self._get_set(set)
        
        if isinstance(dataset, Subset):
            while isinstance(dataset.dataset, (LabelRemapper, DatasetWithIndex, NoisyClassificationDataset)):
                dataset.dataset = dataset.dataset.dataset
        else:
            # unwrap the dataset
            while isinstance(dataset, (LabelRemapper, DatasetWithIndex, NoisyClassificationDataset)):
                dataset = dataset.dataset
        
        
        dataset = NoisyClassificationDataset(
            dataset=dataset,
            dataset_name=self.__class__.__name__,
            noise_rate=noise_rate,
            noise_type=noise_type,
            T_mat=T_mat,
            seed=seed,
            num_classes=len(self.available_classes),
            target_class=target_class,
            available_labels=self.class_subset,
            generator=generator
        )
        
        if self.remap_labels and self.class_subset:
            dataset = LabelRemapper(dataset, self.label_mapping)
        
        dataset = DatasetWithIndex(dataset)
        
        self._set_set(set, dataset)
        
        
    def get_clean_noisy_subsets(self, set='Train'):
        dataset = self._get_set(set)
        
        clean_indices = []
        noisy_indices = []
        for item in dataset:
            if len(item) == 4:
                x, y, idx, is_noisy = item
                if is_noisy:
                    noisy_indices.append(idx)
                else:
                    clean_indices.append(idx)
            else:
                raise RuntimeError('The chosen dataset is not noisy!')
        
        return Subset(dataset, clean_indices), Subset(dataset, noisy_indices)
    
    
    def _init_loaders(self):
        train_dataset = self.load_train_set()
        val_dataset = self.load_validation_set()
        test_dataset = self.load_test_set()

        if self.class_subset:
            train_idxs = [i for i, lbl in enumerate(train_dataset.targets) if lbl in self.class_subset]
            train_dataset = Subset(train_dataset, train_idxs)
            
            test_idxs = [i for i, lbl in enumerate(test_dataset.targets) if lbl in self.class_subset]
            test_dataset = Subset(test_dataset, test_idxs)
        
        if self.subsample_size[1] != -1:
            test_indices = torch.randperm(len(test_dataset), generator=self.generator)[:self.subsample_size[1]]
            test_dataset = Subset(test_dataset, test_indices.tolist())
            
        
        if self.balance_classes and self.class_subset and self.subsample_size[0] != -1:
            train_dataset = self._get_balanced_subset(train_dataset, self.subsample_size[0], self.class_subset, self.generator)
        elif self.subsample_size[0] != -1:
            train_indices = torch.randperm(len(train_dataset), generator=self.generator)[:self.subsample_size[0]]
            train_dataset = Subset(train_dataset, train_indices.tolist())

        heldout_set = None
        if self.heldout_conf:
            train_dataset, heldout_set = self._split_heldout_set(train_dataset)

        
        if val_dataset != None:
            valset = val_dataset
        else:
            if self.valset_ratio > 0.0 and len(train_dataset) > 1:
                trainset, valset = random_split(train_dataset, [self.trainset_ratio, self.valset_ratio], generator=self.generator)
            else:
                trainset, valset = train_dataset, None

        if self.remap_labels and self.class_subset:
            self.label_mapping = {orig: new for new, orig in enumerate(self.class_subset)}
            self.available_classes = sorted(list(self.label_mapping.values()))
            # print(self.available_classes)
            trainset = LabelRemapper(trainset, self.label_mapping)
            if valset: valset = LabelRemapper(valset, self.label_mapping)
            test_dataset = LabelRemapper(test_dataset, self.label_mapping)
            if heldout_set: heldout_set = LabelRemapper(heldout_set, self.label_mapping)

        
        trainset = DatasetWithIndex(train_dataset)
        if valset: valset = DatasetWithIndex(valset)
        test_dataset = DatasetWithIndex(test_dataset)
        if heldout_set: heldout_set = DatasetWithIndex(heldout_set)
        
        self.trainset = trainset
        self.valset = valset
        self.testset = test_dataset
        self.heldout_set = heldout_set
        
        self.train_loader = self._build_dataloader(self.trainset, shuffle=True)
        self.val_loader = self._build_dataloader(self.valset) if self.valset else None
        self.test_loader = self._build_dataloader(self.testset)
        self.heldout_loader = self._build_dataloader(self.heldout_set) if self.heldout_set else None
        
    def _build_dataloader(self, dataset, shuffle=False):
        if not dataset or len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator,
        )
        
    def _get_balanced_subset(self, dataset: Dataset, total_size: int, class_subset: list, generator: torch.Generator) -> Subset:
        num_classes = len(class_subset)
        if total_size == -1 or total_size is None:
             return dataset

        if total_size % num_classes != 0:
            raise ValueError(
                f"For balanced sampling, the subsample size ({total_size}) must be "
                f"perfectly divisible by the number of classes ({num_classes})."
            )
        
        samples_per_class = total_size // num_classes
        
        # This approach is robust to `dataset` being a Subset
        labels = [dataset[i][1] for i in range(len(dataset))]
        indices_by_class = {cls: [] for cls in class_subset}
        for i, label in enumerate(labels):
            if label in indices_by_class:
                indices_by_class[label].append(i)

        final_indices = []
        for class_label in class_subset:
            class_indices = indices_by_class[class_label]
            if len(class_indices) < samples_per_class:
                raise ValueError(
                    f"Cannot sample {samples_per_class} for class {class_label}, "
                    f"as only {len(class_indices)} are available in the filtered dataset."
                )
            
            perm = torch.randperm(len(class_indices), generator=generator)
            selected_indices = [class_indices[i] for i in perm[:samples_per_class]]
            final_indices.extend(selected_indices)
            
        # shuffled_perm = torch.randperm(len(final_indices), generator=generator)
        # shuffled_final_indices = torch.tensor(final_indices)[shuffled_perm].tolist()

        return Subset(dataset, final_indices)
        
    def _split_heldout_set(self, dataset: Dataset):
        """
        Splits a dataset into a training part and a held-out part based on heldout_conf.
        This method performs stratified sampling for tuple configurations.
        """
        indices_in_view = list(range(len(dataset)))
        labels_in_view = [dataset[i][1] for i in indices_in_view]
        
        indices_by_class = {}
        for idx, label in zip(indices_in_view, labels_in_view):
            label_item = label.item() if isinstance(label, torch.Tensor) else label
            if label_item not in indices_by_class:
                indices_by_class[label_item] = []
            indices_by_class[label_item].append(idx)
        
        heldout_view_indices = []
        if isinstance(self.heldout_conf, float):
            # Hold out a portion of *each available class*.
            ratio = self.heldout_conf
            for class_label in self.available_classes:
                if class_label in indices_by_class:
                    class_view_indices = indices_by_class[class_label]
                    num_to_hold = int(len(class_view_indices) * ratio)
                    if num_to_hold > 0:
                        perm = torch.randperm(len(class_view_indices), generator=self.generator)
                        heldout_view_indices.extend([class_view_indices[i] for i in perm[:num_to_hold]])

        elif isinstance(self.heldout_conf, dict):
            # Sanitize keys to be integers, just in case
            safe_conf = {int(k): v for k, v in self.heldout_conf.items()}
            for class_label, ratio in safe_conf.items():
                if class_label in indices_by_class:
                    class_view_indices = indices_by_class[class_label]
                    num_to_hold = int(len(class_view_indices) * ratio)
                    if num_to_hold > 0:
                        perm = torch.randperm(len(class_view_indices), generator=self.generator)
                        heldout_view_indices.extend([class_view_indices[i] for i in perm[:num_to_hold]])
        
        if not heldout_view_indices:
            # If no indices were selected, return the original dataset and an empty set
            return dataset, None

        train_view_indices = [i for i in indices_in_view if i not in heldout_view_indices]
        
        return Subset(dataset, train_view_indices), Subset(dataset, heldout_view_indices)
        
    def _get_set(self, set):
        dataset = None
        if set == 'Train':
            dataset = self.trainset
        elif set == 'Val':
            dataset = self.valset
        elif set == 'Test':
            dataset = self.testset
        elif set == 'Heldout':
            dataset = self.heldout_set
        else:
            raise ValueError('set argument must be one of these values `Train`, `Val`, `Test`, `Heldout`')
        
        return dataset
    
    def _set_set(self, set, new_set):
        if set == 'Train':
            self.trainset = new_set 
            self.train_loader = self._build_dataloader(self.trainset, shuffle=True)
        elif set == 'Val':
            self.valset = new_set
            self.val_loader = self._build_dataloader(self.valset, shuffle=False)
        elif set == 'Test':
            self.testset = new_set 
            self.test_loader = self._build_dataloader(self.testset, shuffle=False)
        elif set == 'Heldout':
            self.heldout_set = new_set
            self.heldout_loader = self._build_dataloader(self.heldout_set, shuffle=False)
            
        else:
            raise ValueError('set argument must be one of these values `Train`, `Val`, `Test`, `Heldout`')