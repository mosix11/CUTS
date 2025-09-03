import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as transforms

import torchmetrics
from torch.amp import autocast

from . import BaseModel
from tqdm import tqdm

from typing import Union, List, Callable



class DinoV3Classifier(BaseModel):
    
    def __init__(
        self,
        pt_weights:str = None,
        num_classes:int = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.pt_weights = pt_weights
        
        self.image_encoder = AutoModel.from_pretrained(pt_weights)
        self.pre_processor = AutoImageProcessor.from_pretrained(pt_weights)
        
        feature_dim = self.image_encoder.config.hidden_size
        self.classifier_head = nn.Linear(feature_dim, num_classes)
        
        
                
    def forward(self, x):
        ftrs = self.image_encoder(x)
        logits = self.classifier_head(ftrs)
        return logits
    
    
    
    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=self.pre_processor.image_mean, std=self.pre_processor.image_std)
        ])
    
    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=self.pre_processor.image_mean, std=self.pre_processor.image_std),
        ])
        
        
    def get_identifier(self):
        return f"dinov3/{self.pt_weights}"