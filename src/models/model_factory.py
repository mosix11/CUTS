import torch
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryF1Score, MulticlassF1Score
from . import FC1, FCN, CNN5
from . import PreActResNet9, PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from . import PostActResNet9, PostActResNet18, PostActResNet34, PostActResNet50, PostActResNet101, PostActResNet152
from . import PostActResNet9_ETD
from . import CNN5_ETD
from . import ViT_Small
from . import TorchvisionModels, TimmModels
from . import OpenClipImageEncoder, OpenClipImageClassifier, OpenClipMultiHeadImageClassifier

from .loss_functions import SupervisedContrastiveLoss, CompoundLoss


def get_metric(metric_name, num_classes):
    if metric_name == 'ACC':
        if num_classes == 2:   
            return BinaryAccuracy()
        else:
            return MulticlassAccuracy(num_classes=num_classes, average='micro')
    elif metric_name == 'F1':
        if num_classes == 2:
            return BinaryF1Score()
        else:
            return MulticlassF1Score(num_classes=num_classes, average='micro')
    else: raise ValueError(f"Invalid metric {metric_name}.")
def create_model(cfg, num_classes=None):
    model_type = cfg.pop('type')
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
    elif loss_fn_type == 'SCL':
        cfg['loss_fn'] = SupervisedContrastiveLoss(
            **loss_fn_cfg
        )
        
    else: raise ValueError(f"Invalid loss function {cfg['loss_fn']['type']}.")
    
    
    if cfg['metrics']:
        metrics = {}
        if num_classes:
            for metric_name in cfg['metrics']:    
                metrics[metric_name] = get_metric(metric_name, num_classes)
        elif 'heads_cfg' in cfg:
            for head_cfg in cfg['heads_cfg']:
                head_name, head_out_dim = head_cfg['head_name'], head_cfg['head_out_dim']
                metrics[head_name] = {}
                for metric_name in cfg['metrics']:
                    metrics[head_name][metric_name] = get_metric(metric_name, num_classes=head_out_dim)   
            
        cfg['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)
    elif model_type == 'fcN':
        model = FCN(**cfg)
        
        
    elif model_type == 'cnn5':
        model = CNN5(**cfg)
        
        
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
        
    elif model_type.startswith('open_clip'):
        model_type = model_type.removeprefix('open_clip_')
        if model_type.startswith('multi_head'):
            model_type = model_type.removeprefix('multi_head_')
            model = OpenClipMultiHeadImageClassifier(
                model_type=model_type,
                **cfg
            )
        elif model_type.startswith('single_head'):
            model_type = model_type.removeprefix('single_head_')
            model = OpenClipImageClassifier(
                model_type=model_type,
                **cfg
            )
        elif model_type.startswith('encoder'):
            model_type = model_type.removeprefix('encoder_')
            model = OpenClipImageEncoder(
                model_type=model_type,
                **cfg
            )
    
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model