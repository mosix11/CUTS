from . import CIFAR10, CIFAR100, MNIST

def create_dataset(cfg, augmentations=None):
    dataset_name = cfg['dataset'].pop('name')
    cfg['dataset']['augmentations'] = augmentations if augmentations else []
    
    if dataset_name == 'mnist':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = MNIST(
            **cfg['dataset']
        )
    elif dataset_name == 'cifar10':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg['dataset']
        )
    elif dataset_name == 'cifar100':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR100(
            **cfg['dataset']
        )
    elif dataset_name == 'mog':
        pass
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes