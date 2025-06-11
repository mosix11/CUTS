import torch
import torchmetrics
from . import FC1, FCN, CNN5, make_resnet18k

def create_model(cfg, num_classes):
    model_type = cfg['model'].pop('type')
    loss_fn = cfg['model']['loss_fn']
    if loss_fn == 'MSE':
        cfg['model']['loss_fn'] = torch.nn.MSELoss()
    elif loss_fn == 'MSE-NR':
        cfg['model']['loss_fn'] = torch.nn.MSELoss(reduction='none')
    elif loss_fn == 'CE':
        cfg['model']['loss_fn'] = torch.nn.CrossEntropyLoss()
    elif loss_fn == 'CE-NR':
        cfg['model']['loss_fn'] = torch.nn.CrossEntropyLoss(reduction='none')
    else: raise ValueError(f"Invalid loss function {cfg['model']['loss_fn']}.")
    
    
    if cfg['model']['metrics']:
        metrics = {}
        for metric_name in cfg['model']['metrics']:
            if metric_name == 'ACC':
                metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            elif metric_name == 'F1':
                metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
            else: raise ValueError(f"Invalid metric {metric_name}.")
        cfg['model']['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)
    elif model_type == 'fcN':
        model = FCN(**cfg)
    elif model_type == 'cnn5':
        model = CNN5(**cfg['model'])
    elif model_type == 'resnet18k':
        model = make_resnet18k(**cfg)
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model