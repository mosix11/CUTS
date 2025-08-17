
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
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
                        
                                      
                        
                        
                        
class OpenClipMultiHeadImageClassifier(BaseClassificationModel):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        heads_cfg:dict = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        self.image_encoder = OpenClipImageEncoder(model_name=model_type, pt_weights=pt_weights, keep_lang=False)
        
        self.classifier_heads = nn.ModuleDict()
        for head_name, head_out_dim in heads_cfg.items():
            self.classifier_heads[head_name] = nn.Linear(self.image_encoder.feature_dim, head_out_dim)
        
        # By default, the first head is activated
        self.active_head = next(iter(self.classifier_heads.keys()))

    
    
    def forward(self, x):
        ftrs = self.image_encoder(x)
        # normalize feature
        # ftrs = ftrs / ftrs.norm(dim=-1, keepdim=True)
        logits = self.classifier_heads[self.active_head](ftrs)
        return logits
    
    
    def acitve_head(self, head_name:str):
        if head_name not in self.classifier_heads:
            raise ValueError('The specified head name is not in the classifier heads.')
        self.active_head = head_name
    
    def get_train_transforms(self):
        return self.image_encoder.train_preprocess
    
    def get_val_transforms(self):
        return self.image_encoder.val_preprocess
    
    
    def load_heads(self, state_dicts):
        for head_name, state_dict in state_dicts.items():
            if head_name not in self.classifier_heads:
                raise ValueError('The specified head name is not in the classifier heads.')
            else:
                self.classifier_heads[head_name].load_state_dict(state_dict)
        
    def freeze_head(self, head_name:Union[str, List[str]]):
        if isinstance(head_name, List):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(False)
        elif isinstance(head_name, str):
            self.classifier_heads[hn].requires_grad_(False)
        
    def unfreeze_head(self, head_name:Union[str, List[str]]):
        if isinstance(head_name, List):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(True)
        elif isinstance(head_name, str):
            self.classifier_heads[hn].requires_grad_(True)
            
    
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