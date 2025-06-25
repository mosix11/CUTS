from . import CIFAR10

def create_dataset(cfg, augmentations=None, phase:str='pretraining'):
    cfg['dataset']['batch_size'] = cfg['trainer'][phase]['batch_size']
    del cfg['trainer'][phase]['batch_size']
    dataset_name = cfg['dataset'].pop('name')
    cfg['dataset']['augmentations'] = augmentations if augmentations else []
    
    if dataset_name == 'mnist':
        pass
    elif dataset_name == 'cifar10':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg['dataset']
        )
    elif dataset_name == 'cifar100':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg['dataset']
        )
    elif dataset_name == 'mog':
        pass
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes