import torch
import torchvision as tv
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torch.utils.data import Dataset, ConcatDataset
from .base_classification_dataset import BaseClassificationDataset
from .dataset_wrappers import DatasetWithIndex, LabelRemapper, NoisyClassificationDataset, BinarizedClassificationDataset

import torch.distributed as dist

import os
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List, Union, Dict
import openxlab
from openxlab.dataset import get as openxlab_get
from openxlab.dataset import download as openxlab_download
from .utils import extract_zip, extract_tar_gz, extract_tar
import dotenv
import shutil

from PIL import Image
1

class Clothing1MDataset(Dataset):
    """
    PyTorch Dataset for Clothing1M splits organized under root_dir.
    Expects subfolders q0000..q0013 each containing image files.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.startswith('q'):
                # drop leading 'q' and parse class idx
                label = int(class_dir.name[1:])
                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        self.samples.append((img_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class Clothing1M(BaseClassificationDataset):
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        dotenv_path: Path = Path("./.env"),
        img_size: Union[tuple, list] = (224, 224),
        grayscale: bool = False,
        normalize_imgs: bool = False,
        flatten: bool = False,
        augmentations: Union[list, None] = None,
        train_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        val_transforms: Union[tv.transforms.Compose, transforms.Compose] = None,
        **kwargs
    ) -> None:
        
        if dotenv_path.exists():
            dotenv.load_dotenv('.env')
        
        
        
        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir / 'Clothing1M'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_dir = dataset_dir
    
    
        if self.is_distributed():
            if self.is_node_leader():
                self._download_dataset()
            dist.barrier()
        else:
            self._download_dataset()
         
        
        
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten
        self.augmentations = [] if augmentations == None else augmentations
        
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        
        if (train_transforms or val_transforms) and (augmentations != None):
            raise ValueError('You should either pass augmentations, or train and validation transforms.')
        
        super().__init__(
            dataset_name='Clothing1M',
            dataset_dir=dataset_dir,
            num_classes=14,
            **kwargs,  
        )
        


    def load_train_set(self):
        self.train_transforms = self.get_transforms(train=True)
        return Clothing1MDataset(root_dir=self.train_noisy_dir, transform=self.train_transforms)
    
    def load_validation_set(self):
        # TODO change this back
        # self.val_transforms = self.get_transforms(train=False)
        self.val_transforms = self.get_transforms(train=True)
        valset = Clothing1MDataset(root_dir=self.val_dir, transform=self.val_transforms)
        # clean_trainset = Clothing1MDataset(root_dir=self.train_clean_dir, transform=self.val_transforms)
        # return ConcatDataset([valset, clean_trainset])
        return valset
    
    def load_test_set(self):
        self.val_transforms = self.get_transforms(train=False)
        return Clothing1MDataset(root_dir=self.test_dir, transform=self.val_transforms)

    def get_transforms(self, train=True):
        if self._train_transforms and train:
            return self._train_transforms
        elif self._val_transforms and not train:
            return self._val_transforms
        
        trnsfrms = []
        if train:
            trnsfrms.append(transforms.RandomResizedCrop((224, 224)))
        else:
            trnsfrms.append(transforms.CenterCrop((224, 224)))
            
        
        if self.grayscale:
            trnsfrms.append(transforms.Grayscale(num_output_channels=1))
        if len(self.augmentations) > 0 and train:
            trnsfrms.extend(self.augmentations)
        trnsfrms.extend([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        if self.normalize_imgs:
            mean, std = ((0.5,), (0.5,)) if self.grayscale else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)


    def get_class_names(self):
        return ["T-Shirt", "Shirt", "Knitwear", "Chiffon", "Sweater", "Hoodie", "Windbreaker", "Jacket", "Downcoat", "Suit", "Shawl", "Dress", "Vest", "Underwear"]

    def get_identifier(self):
        identifier = 'clothing1M|'
        # identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
    
    def _download_dataset(self):
        
        self.train_noisy_dir = self.dataset_dir / 'noisy_train'
        self.train_clean_dir = self.dataset_dir / 'clean_train'
        self.val_dir = self.dataset_dir / 'val'
        self.test_dir = self.dataset_dir / 'test'
        
        if self.train_clean_dir.exists() and self.train_noisy_dir.exists() and self.val_dir.exists() and self.test_dir.exists():
            return
        
        extraction_dir = self.dataset_dir / 'extracted_files'
        extraction_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = self.dataset_dir / 'OpenDataLab___Clothing1M/raw/Clothing1M.tar.gz'
            
        OPENXLAB_AK = os.getenv("OPENXLAB_AK")
        OPENXLAB_SK = os.getenv("OPENXLAB_SK")
        
        if not zip_file_path.exists():
            openxlab.login(ak=OPENXLAB_AK, sk=OPENXLAB_SK) 
            openxlab_get(dataset_repo='OpenDataLab/Clothing1M', target_path=str(self.dataset_dir.absolute())) 
        
        extract_tar_gz(self.dataset_dir / 'OpenDataLab___Clothing1M/raw/Clothing1M.tar.gz', extraction_dir)
        self._extract_and_organize_files(extraction_dir / 'Clothing1M/clothing1M')
        
        shutil.move(extraction_dir / 'Clothing1M/clothing1M/clean_train', self.dataset_dir)
        shutil.move(extraction_dir / 'Clothing1M/clothing1M/noisy_train', self.dataset_dir)
        shutil.move(extraction_dir / 'Clothing1M/clothing1M/val', self.dataset_dir)
        shutil.move(extraction_dir / 'Clothing1M/clothing1M/test', self.dataset_dir)
        
        # moving annotations
        self.annotations_dir = self.dataset_dir / 'annotations'
        self.annotations_dir.mkdir(exist_ok=True, parents=True)
        for txt_file in extraction_dir.joinpath('Clothing1M/clothing1M').glob("*.txt"):
            shutil.move(str(txt_file), self.annotations_dir / txt_file.name)

        shutil.rmtree(extraction_dir, ignore_errors=True)
        # shutil.rmtree(self.dataset_dir / 'OpenDataLab___Clothing1M', ignore_errors=True)

    def _extract_and_organize_files(self, root_dir:Path):
        """
        Prepares and organizes the Clothing1M dataset files.
        Steps:
        - Creates directory structure (clean_train, val, test, noisy_train with subfolders q0000..q0013)
        - Unzips annotations.zip and untars images/0.tar..9.tar
        - Builds key-value files for clean and noisy splits
        - Moves/copies files into class-specific directories
        - Cleans up temporary files

        Args:
            root_dir: Path to the base dataset directory containing:
            - annotations.zip
            - images/ directory or tar archives
            - clean_label_kv.txt, clean_*.txt, noisy_label_kv.txt, noisy_train_key_list.txt
        """
        root = root_dir

        # 1. Create directory structure
        print("Creating directory structure...")
        splits = ['clean_train', 'val', 'test', 'noisy_train']
        for split in splits:
            split_dir = root / split
            split_dir.mkdir(exist_ok=True)
            for i in range(14):  # 0..13
                sub = f"q00{i}" if i > 9 else f"q000{i}"
                (split_dir / sub).mkdir(parents=True, exist_ok=True)

        # 2. Unzip and untar
        ann_zip = root / 'annotations.zip'
        if ann_zip.exists():
            print("Unzipping annotations...")
            extract_zip(ann_zip, root)
        img_dir = root / 'images'
        if img_dir.exists():
            print("Extracting image tar files...")
            for i in range(10):
                tar_path = img_dir / f"{i}.tar"
                if tar_path.exists():
                    extract_tar(tar_path, img_dir)

        # 3. Helper logic (generate clean_*_kv.txt and noisy_train_kv.txt)
        print("Generating key-value mapping files...")
        # Load clean labels
        clean_label_path = root / 'clean_label_kv.txt'
        clean_map = {}
        with open(clean_label_path) as f:
            for line in f:
                img, cls = line.strip().split()
                clean_map[img] = cls

        def build_kv(keys_file, out_file, filter_exclude=None):
            d = {}
            with open(root / keys_file) as f:
                for line in f:
                    key = line.strip()
                    d[key] = clean_map[key]
            # write
            with open(root / out_file, 'w') as fo:
                for k, v in d.items():
                    if not filter_exclude or k not in filter_exclude:
                        fo.write(f"{k} {v}\n")
            return set(d.keys())

        clean_val = build_kv('clean_val_key_list.txt', 'clean_val_kv.txt')
        clean_test = build_kv('clean_test_key_list.txt', 'clean_test_kv.txt')
        clean_train = build_kv('clean_train_key_list.txt', 'clean_train_kv.txt', filter_exclude=clean_val.union(clean_test))

        # Noisy labels
        noisy_label_path = root / 'noisy_label_kv.txt'
        noisy_map = {}
        with open(noisy_label_path) as f:
            for line in f:
                img, cls = line.strip().split()
                noisy_map[img] = cls
        noisy_keys = []
        with open(root / 'noisy_train_key_list.txt') as f:
            noisy_keys = [line.strip() for line in f]
        with open(root / 'noisy_train_kv.txt', 'w') as fo:
            for key in noisy_keys:
                if key not in clean_val and key not in clean_test:
                    fo.write(f"{key} {noisy_map[key]}\n")

        # 4. Move/Copy files (flatten image paths into class folders)
        print("Organizing dataset into class directories...")
        def dispatch(kv_file, src_base, dst_base, action='mv'):
            with open(root / kv_file) as f:
                for line in f:
                    img, cls = line.strip().split()
                    subdir = f"q00{cls}" if int(cls) > 9 else f"q000{int(cls)}"
                    src = root / src_base / img
                    # use only the filename (flatten nested dirs)
                    filename = Path(img).name
                    dst = root / dst_base / subdir / filename
                    # ensure destination parent exists
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    if action == 'mv':
                        shutil.move(str(src), str(dst))
                    else:
                        shutil.copy(str(src), str(dst))

        dispatch('clean_val_kv.txt', '.', 'val', 'mv')
        dispatch('clean_test_kv.txt', '.', 'test', 'mv')
        dispatch('clean_train_kv.txt', '.', 'clean_train', 'cp')
        dispatch('noisy_train_kv.txt', '.', 'noisy_train', 'cp')

        # 5. Cleanup
        print("Cleaning up temporary files...")
        shutil.rmtree(root / 'images', ignore_errors=True)
        try:
            (root / 'annotations.zip').unlink()
        except FileNotFoundError:
            pass

        print("Dataset preparation complete.")