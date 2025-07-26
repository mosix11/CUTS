from . import CIFAR10, CIFAR100, MNIST, FashionMNIST
import copy

def create_dataset(cfg, augmentations=None):
    cfg_cpy = copy.deepcopy(cfg)
    dataset_name = cfg_cpy['dataset'].pop('name')
    cfg_cpy['dataset']['augmentations'] = augmentations if augmentations else []
    
    if dataset_name == 'mnist':
        num_classes = cfg_cpy['dataset'].pop('num_classes')
        dataset = MNIST(
            **cfg_cpy['dataset']
        )
    elif dataset_name == 'fashion_mnist':
        num_classes = cfg_cpy['dataset'].pop('num_classes')
        dataset = FashionMNIST(
            **cfg_cpy['dataset']
        )
    elif dataset_name == 'cifar10':
        num_classes = cfg_cpy['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg_cpy['dataset']
        )
    elif dataset_name == 'cifar100':
        num_classes = cfg_cpy['dataset'].pop('num_classes')
        dataset = CIFAR100(
            **cfg_cpy['dataset']
        )
    elif dataset_name == 'mog':
        pass
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes