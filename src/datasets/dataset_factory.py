from . import CIFAR10, CIFAR100, MNIST, FashionMNIST, MoGSynthetic, DummyClassificationDataset
import copy

def create_dataset(cfg, augmentations=None):
    cfg_cpy = copy.deepcopy(cfg)
    dataset_name = cfg_cpy['dataset'].pop('name')
    
    if augmentations:
        cfg_cpy['dataset']['augmentations'] = augmentations
    
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
        num_classes = cfg_cpy['dataset']['num_classes']
        dataset = MoGSynthetic(
            **cfg_cpy['dataset']
        )
        
    elif dataset_name == 'dummy_class':
        num_classes = 10
        dataset = DummyClassificationDataset(
            **cfg_cpy['dataset']
        )
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes