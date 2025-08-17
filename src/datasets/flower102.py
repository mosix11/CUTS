import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from .base_classification_dataset import BaseClassificationDataset
from typing import Tuple, List, Union, Dict
from pathlib import Path

class Flowers102(BaseClassificationDataset):
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
        dataset_dir = data_dir / 'Flowers102'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        super().__init__(
            dataset_name='Flowers102',
            dataset_dir=dataset_dir,
            num_classes=102,
            **kwargs,  
        )


    def load_train_set(self):
        return datasets.Flowers102(root=self.dataset_dir, split="train", transform=self.get_transforms(train=True), download=True)
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        return datasets.Flowers102(root=self.dataset_dir, split="test", transform=self.get_transforms(train=False), download=True)

    def get_transforms(self, train=True):
        if self.train_transforms and train:
            return self.train_transforms
        elif self.val_transforms and not train:
            return self.val_transforms
        
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
            mean, std = ((0.5,), (0.5,)) if self.grayscale else ((0.4340, 0.3857, 0.2965), (0.2948, 0.2435, 0.2546))
            trnsfrms.append(transforms.Normalize(mean, std))
        if self.flatten:
            trnsfrms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        return transforms.Compose(trnsfrms)


    def get_class_names(self):
        return [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink and yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "air plant",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]

    def get_identifier(self):
        identifier = 'food101|'
        # identifier += f'ln{self.label_noise}|'
        identifier += 'aug|' if len(self.augmentations) > 0 else 'noaug|'
        identifier += f'subsample{self.subsample_size}' if self.subsample_size != (-1, -1) else 'full'
        return identifier
    
    

    



            