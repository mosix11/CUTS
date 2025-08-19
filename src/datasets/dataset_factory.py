from . import CIFAR10, CIFAR100, MNIST, FashionMNIST, MoGSynthetic, DummyClassificationDataset, Clothing1M
from . import KMNIST, Food101, Flowers102, Country211, EMNIST, OxfordIIITPet, PCAM, SVHN
import copy

def create_dataset(cfg, augmentations=None):
    cfg_cpy = copy.deepcopy(cfg)
    dataset_name = cfg_cpy.pop('name')
    
    if augmentations:
        cfg_cpy['augmentations'] = augmentations
    
    if dataset_name == 'mnist':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = MNIST(
            **cfg_cpy
        )
    elif dataset_name == 'fashion_mnist':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = FashionMNIST(
            **cfg_cpy
        )
    elif dataset_name == 'cifar10':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = CIFAR10(
            **cfg_cpy
        )
    elif dataset_name == 'cifar100':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = CIFAR100(
            **cfg_cpy
        )
    elif dataset_name == 'mog':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = MoGSynthetic(
            **cfg_cpy
        )
    elif dataset_name == 'clothing1M':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = Clothing1M(
            **cfg_cpy
        )
    elif dataset_name == 'kmnist':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = KMNIST(
            **cfg_cpy
        )
    elif dataset_name == 'food101':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = Food101(
            **cfg_cpy
        )
    elif dataset_name == 'flowers102':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = Flowers102(
            **cfg_cpy
        )
    elif dataset_name == 'svhn':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = SVHN(
            **cfg_cpy
        )
    elif dataset_name == 'country211':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = Country211(
            **cfg_cpy
        )
    elif dataset_name == 'emnist':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = EMNIST(
            **cfg_cpy
        )
    elif dataset_name == 'oxfordpet':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = OxfordIIITPet(
            **cfg_cpy
        )
    elif dataset_name == 'pcam':
        num_classes = cfg_cpy.pop('num_classes')
        dataset = PCAM(
            **cfg_cpy
        )
    elif dataset_name == 'dummy_class':
        num_classes = 10
        dataset = DummyClassificationDataset(
            **cfg_cpy
        )
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes