
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from . import BaseClassificationModel

from pathlib import Path

from typing import Union, List

class OpenClipImageEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name:str,
        pt_weights:str = None,
        keep_lang:bool = False
    ):
        super().__init__()

        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(model_name, pretrained=pt_weights)
        
        dim = None
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            dim = self.model.visual.output_dim
        elif hasattr(self.model, "embed_dim"):
            dim = self.model.embed_dim
        elif hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            dim = self.model.text_projection.shape[-1]
        if dim is None:
            with torch.no_grad():
                H = W = getattr(self.model.visual, "image_size", 224)
                dummy = torch.zeros(1, 3, H, W)
                dim = self.model.encode_image(dummy).shape[-1]
        self.feature_dim = int(dim)

        if not keep_lang:
            for attr in ["transformer","token_embedding","positional_embedding",
                         "ln_final","text_projection","attn_mask","temp_attn_mask","txt_attn_mask"]:
                if hasattr(self.model, attr):
                    delattr(self.model, attr)
            setattr(self.model, "encode_text", None)

    def forward(self, images):
        return self.model.encode_image(images)


    # @classmethod
    # def load(cls, model_name:str, filename:Path):
    #     print(f"Loading image encoder from {filename}")

    #     state_dict = torch.load(filename, map_location="cpu")

    #     model = cls(model_name)
    #     model.load_state_dict(state_dict)
    #     return model

class OpenClipImageClassifier(BaseClassificationModel):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        num_classes:int = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        self.image_encoder = OpenClipImageEncoder(model_name=model_type, pt_weights=pt_weights, keep_lang=False)
        
        self.classifier_head = nn.Linear(self.image_encoder.feature_dim, num_classes)
        

    
    def forward(self, x):
        ftrs = self.image_encoder(x)
        # normalize feature
        # ftrs = ftrs / ftrs.norm(dim=-1, keepdim=True)
        logits = self.classifier_head(ftrs)
        return logits
    
    
    def get_identifier(self):
        return 'Open Clip Image Classifer' + self.model_type
    
    def get_train_transforms(self):
        return self.image_encoder.train_preprocess
    
    def get_val_transforms(self):
        return self.image_encoder.val_preprocess
    
    def get_head_weights(self):
        return self.classifier_head.state_dict()
        
    def load_head(self, state_dict):
        self.classifier_head.load_state_dict(state_dict)
        
    def freeze_head(self):
        self.classifier_head.requires_grad_(False)
        
    def unfreeze_head(self):
        self.classifier_head.requires_grad_(True)
    
    def freeze_encoder(self):
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self, top_n_blocks=None):
        if top_n_blocks is None:
            for p in self.image_encoder.parameters():
                p.requires_grad_(True)
        else:
            # works for ViTs with .visual.transformer.resblocks
            blocks = getattr(self.image_encoder.model.visual.transformer, "resblocks", None)
            if blocks is None:
                for p in self.image_encoder.parameters():
                    p.requires_grad_(True)
                return
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            for blk in blocks[-top_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            # always unfreeze final norm/proj
            for attr in ["ln_post","proj","ln_pre"]:
                mod = getattr(self.image_encoder.model.visual, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)
                        
                                      
                        
                        
from torch.amp import autocast
                 
class OpenClipMultiHeadImageClassifier(nn.Module):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        heads_cfg:list = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__()
        
        if loss_fn is None:
            raise RuntimeError('The loss function must be specified for training/validation.')
        self.loss_fn = loss_fn
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for head_cfg in heads_cfg:
                head_name, head_out_dim = head_cfg['head_name'], head_cfg['head_out_dim']
                self.metrics[head_name] = nn.ModuleDict()
                for name, metric_instance in metrics[head_name].items():
                    if not isinstance(metric_instance, torchmetrics.Metric):
                        raise TypeError(f"Metric '{name}' must be an instance of torchmetrics.Metric.")
                    self.metrics[head_name][name] = metric_instance
                
        
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        self.image_encoder = OpenClipImageEncoder(model_name=model_type, pt_weights=pt_weights, keep_lang=False)
        
        self.classifier_heads = nn.ModuleDict()
        for head_cfg in heads_cfg:
            head_name, head_out_dim = head_cfg['head_name'], head_cfg['head_out_dim']
            self.classifier_heads[head_name] = nn.Linear(self.image_encoder.feature_dim, head_out_dim)
        
        # By default, the first head is activated
        self.active_head = next(iter(self.classifier_heads.keys()))

    
    def forward(self, x):
        ftrs = self.image_encoder(x)
        # normalize feature
        # ftrs = ftrs / ftrs.norm(dim=-1, keepdim=True)
        logits = self.classifier_heads[self.active_head](ftrs)
        return logits
    
    
    def training_step(self, x, y, use_amp=False, return_preds=False):
        """Performs a single training step."""
        with autocast('cuda', enabled=use_amp):
            preds = self(x) 
            loss = self.loss_fn(preds, y)

        if self.metrics:
            for name, metric in self.metrics[self.active_head].items():
                metric.update(preds.detach(), y.detach()) 

        if return_preds:
            return loss, preds
        else:
            return loss


    @torch.no_grad()
    def validation_step(self, x, y, use_amp=False, return_preds=False):
        """Performs a single validation step (no gradient computation)."""
        
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            loss = self.loss_fn(preds, y)

        if self.metrics:
            for name, metric in self.metrics[self.active_head].items():
                metric.update(preds.detach(), y.detach())

        if return_preds:
            return loss, preds
        else:
            return loss

    @torch.no_grad()
    def predict(self, x):
        """Performs inference (prediction) without gradient computation."""
        preds = self(x)
        return preds

    def compute_metrics(self):
        """Computes and returns the current metric results."""
        results = {}
        if self.metrics:
            for name, metric in self.metrics[self.active_head].items():
                results[name] = metric.compute().cpu().item()
        return results

    def reset_metrics(self):
        """Resets all tracked metrics."""
        if self.metrics:
            for name, metric in self.metrics[self.active_head].items():
                metric.reset()
    
    def activate_head(self, head_name:str):
        if head_name not in self.classifier_heads:
            raise ValueError('The specified head name is not in the classifier heads.')
        self.active_head = head_name
        
    def get_active_head(self):
        return self.classifier_heads[self.activate_head]
    
    def get_train_transforms(self):
        return self.image_encoder.train_preprocess
    
    def get_val_transforms(self):
        return self.image_encoder.val_preprocess
    
    
    def get_head_weights(self, head_name):
        return self.classifier_heads[head_name].state_dict()
    
    def get_encoder_weights(self):
        return self.image_encoder.state_dict()
    
    def load_heads(self, state_dicts):
        for head_name, state_dict in state_dicts.items():
            if head_name not in self.classifier_heads:
                raise ValueError('The specified head name is not in the classifier heads.')
            else:
                self.classifier_heads[head_name].load_state_dict(state_dict)
        
    def load_encoder(self, state_dict):
        self.image_encoder.load_state_dict(state_dict)
        
    def freeze_head(self, head_name:Union[str, List[str]]):
        if isinstance(head_name, List):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(False)
        elif isinstance(head_name, str):
            self.classifier_heads[hn].requires_grad_(False)
    
    def freeze_all_heads(self):
        for _, cls_head in self.classifier_heads.items():
            cls_head.requires_grad_(False)
        
    def unfreeze_head(self, head_name:Union[str, List[str]]):
        if isinstance(head_name, List):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(True)
        elif isinstance(head_name, str):
            self.classifier_heads[hn].requires_grad_(True)
            
    def unfreeze_all_heads(self):
        for _, cls_head in self.classifier_heads.items():
            cls_head.requires_grad_(True)
            
    
    def freeze_encoder(self):
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self, top_n_blocks=None):
        if top_n_blocks is None:
            for p in self.image_encoder.parameters():
                p.requires_grad_(True)
        else:
            # works for ViTs with .visual.transformer.resblocks
            blocks = getattr(self.image_encoder.model.visual.transformer, "resblocks", None)
            if blocks is None:
                for p in self.image_encoder.parameters():
                    p.requires_grad_(True)
                return
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            for blk in blocks[-top_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            # always unfreeze final norm/proj
            for attr in ["ln_post","proj","ln_pre"]:
                mod = getattr(self.image_encoder.model.visual, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)
                        
                     
                        
    def get_identifier(self):
        return 'Open Clip Image Classifer' + self.model_type
    
    def _count_trainable_parameters(self):
        """Counts and returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)