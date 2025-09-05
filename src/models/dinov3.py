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
from huggingface_hub import login
import dotenv
import os
from pathlib import Path

class DinoV3Classifier(BaseModel):
    
    def __init__(
        self,
        pt_weights:str = None,
        num_classes:int = None,
        loss_fn:nn.Module = None,
        metrics:dict = None,
        dotenv_path: Path = Path("./.env"),
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        
        if dotenv_path.exists():
            dotenv.load_dotenv('.env')
        else:
            raise RuntimeError('Please put the Hugging Face token inside .env file.')
        login(token=os.getenv("HF_TOKEN"))
        
        self.pt_weights = pt_weights
        print(pt_weights)
        
        self.image_encoder = AutoModel.from_pretrained(pt_weights)
        self.pre_processor = AutoImageProcessor.from_pretrained(pt_weights)
        
        feature_dim = self.image_encoder.config.hidden_size
        self.classifier_head = nn.Linear(feature_dim, num_classes)
        
                
    # def forward(self, x):
    #     ftrs = self.image_encoder(x)
    #     logits = self.classifier_head(ftrs)
    #     return logits
    
    def forward(self, x: torch.Tensor):
        outputs = self.image_encoder(pixel_values=x)

        feats = getattr(outputs, "pooler_output", None)
        if feats is None:
            feats = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier_head(feats)
        return logits
    
    
    def get_train_transforms(self):
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(
                        size=(224, 224),
                        scale=(0.9, 1.0),
                        ratio=(0.75, 1.3333),
                        antialias=True),
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