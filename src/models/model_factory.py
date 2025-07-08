import torch
import torchmetrics
from . import FC1, FCN, CNN5, CNN5_NoNorm, CNN5_GN
from . import PreActResNet9, PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from . import PostActResNet9, PostActResNet18, PostActResNet34, PostActResNet50, PostActResNet101, PostActResNet152
from . import PostActResNet9_ETD
from . import CNN5_ETD
from . import ViT_Small
from . import TorchvisionModels, TimmModels

def create_model(cfg, num_classes):
    model_type = cfg.pop('type')
    loss_fn = cfg['loss_fn']
    if loss_fn == 'MSE':
        cfg['loss_fn'] = torch.nn.MSELoss()
    elif loss_fn == 'MSE-NR':
        cfg['loss_fn'] = torch.nn.MSELoss(reduction='none')
    elif loss_fn == 'CE':
        cfg['loss_fn'] = torch.nn.CrossEntropyLoss()
    elif loss_fn == 'CE-NR':
        cfg['loss_fn'] = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_fn == 'BCE':
        cfg['loss_fn'] = torch.nn.BCEWithLogitsLoss()
    else: raise ValueError(f"Invalid loss function {cfg['loss_fn']}.")
    
    
    if cfg['metrics']:
        metrics = {}
        for metric_name in cfg['metrics']:
            if metric_name == 'ACC':
                if num_classes == 1:   
                    metrics[metric_name] = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
                else:
                    metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            elif metric_name == 'F1':
                if num_classes == 1:
                    metrics[metric_name] = torchmetrics.F1Score(task="binary", num_classes=num_classes)
                else:
                    metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
                    
            else: raise ValueError(f"Invalid metric {metric_name}.")
        cfg['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)
    elif model_type == 'fcN':
        model = FCN(**cfg)
        
        
    elif model_type == 'cnn5':
        model = CNN5(**cfg)
    elif model_type == 'cnn5_nonorm':
        model = CNN5_NoNorm(**cfg)
    elif model_type == 'cnn5_gn':
        model = CNN5_GN(**cfg)
        
        
    elif model_type == 'resnet9v2':
        model = PreActResNet9(**cfg)
    elif model_type == 'resnet18v2':
        model = PreActResNet18(**cfg)
    elif model_type == 'resnet34v2':
        model = PreActResNet34(**cfg)
    elif model_type == 'resnet50v2':
        model = PreActResNet50(**cfg)
    elif model_type == 'resnet101v2':
        model = PreActResNet101(**cfg)
    elif model_type == 'resnet152v2':
        model = PreActResNet152(**cfg)
        
        
    elif model_type == 'resnet9v1':
        model = PostActResNet9(**cfg)
    elif model_type == 'resnet18v1':
        model = PostActResNet18(**cfg)
    elif model_type == 'resnet34v1':
        model = PostActResNet34(**cfg)
    elif model_type == 'resnet50v1':
        model = PostActResNet50(**cfg)
    elif model_type == 'resnet101v1':
        model = PostActResNet101(**cfg)
    elif model_type == 'resnet152v1':
        model = PostActResNet152(**cfg)
        
    elif model_type == 'vit_small':
        model = ViT_Small(**cfg)
    elif model_type == 'resnet9v1_etd':
        model = PostActResNet9_ETD(**cfg)
        
    elif model_type == 'cnn5_etd':
        model = CNN5_ETD(**cfg)
        
    elif model_type.startswith('torchvision'): 
        model_type = model_type.removeprefix('torchvision_')
        model = TorchvisionModels(
            model_type=model_type,
            **cfg
        )
    elif model_type.startswith('timm'):
        model_type = model_type.removeprefix('timm_')
        model = TimmModels(
            model_type=model_type,
            **cfg
        )
    
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model