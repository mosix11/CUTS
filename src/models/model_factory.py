import torch
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryF1Score, MulticlassF1Score
from . import FC1
from . import PreActResNet9, PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from . import PostActResNet9, PostActResNet18, PostActResNet34, PostActResNet50, PostActResNet101, PostActResNet152

from . import TorchvisionModels, TorchvisionModelsSAP, DinoV3Classifier
from . import CLIPMultiHeadImageClassifier, OpenClipMultiHeadImageClassifier



import os
import copy

def get_metric(metric_name, num_classes):
    if metric_name == 'ACC':
        if num_classes == 2:   
            return BinaryAccuracy(sync_on_compute=_is_distributed())
        else:
            return MulticlassAccuracy(num_classes=num_classes, average='micro', sync_on_compute=_is_distributed())
    elif metric_name == 'F1':
        if num_classes == 2:
            return BinaryF1Score(sync_on_compute=_is_distributed())
        else:
            return MulticlassF1Score(num_classes=num_classes, average='micro', sync_on_compute=_is_distributed())
    else: raise ValueError(f"Invalid metric {metric_name}.")
    
def create_model(cfg_orig, num_classes=None):
    cfg = copy.deepcopy(cfg_orig)
    model_type = cfg.pop('type')
    if not num_classes:
        num_classes = cfg.get('num_classes', None)
    loss_fn_cfg = cfg['loss_fn']
    loss_fn_type = loss_fn_cfg.pop('type')
    if loss_fn_type == 'MSE':
        cfg['loss_fn'] = torch.nn.MSELoss()
    elif loss_fn_type == 'MSE-NR':
        cfg['loss_fn'] = torch.nn.MSELoss(reduction='none')
    elif loss_fn_type == 'CE':
        cfg['loss_fn'] = torch.nn.CrossEntropyLoss()
    elif loss_fn_type == 'CE-NR':
        cfg['loss_fn'] = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_fn_type == 'BCE':
        cfg['loss_fn'] = torch.nn.BCEWithLogitsLoss()

        
    else: raise ValueError(f"Invalid loss function {cfg['loss_fn']['type']}.")
    
    
    if cfg['metrics']:
        metrics = {}
        if num_classes:
            for metric_name in cfg['metrics']:    
                metrics[metric_name] = get_metric(metric_name, num_classes)
        elif 'datasets_cfgs' in cfg:
            for dataset_name, class_names in cfg['datasets_cfgs'].items():
                metrics[dataset_name] = {}
                for metric_name in cfg['metrics']:
                    metrics[dataset_name][metric_name] = get_metric(metric_name, num_classes=len(class_names))   
            
        cfg['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)

        
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
        
        
    elif model_type == 'dinov3':
        model = DinoV3Classifier(**cfg)

        
    elif model_type.startswith('torchvisionsap'): 
        model_type = model_type.removeprefix('torchvisionsap_')
        model = TorchvisionModelsSAP(
            model_type=model_type,
            **cfg
        )
        
    elif model_type.startswith('torchvision'): 
        model_type = model_type.removeprefix('torchvision_')
        model = TorchvisionModels(
            model_type=model_type,
            **cfg
        )
    
        
        
    elif model_type.startswith('open_clip'):
        model_type = model_type.removeprefix('open_clip_')
        if model_type.startswith('multihead'):
            model_type = model_type.removeprefix('multihead_')
            model = OpenClipMultiHeadImageClassifier(
                model_type=model_type,
                **cfg
            )
    elif model_type.startswith('clip'):
        model_type = model_type.removeprefix('clip_')
        if model_type.startswith('multihead'):
            model_type = model_type.removeprefix('multihead_')
            model = CLIPMultiHeadImageClassifier(
                model_type=model_type,
                **cfg
            )
    
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model



def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))