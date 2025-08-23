
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.amp import autocast
from . import BaseModel
from ..datasets import get_clip_templates
from ..trainers import utils as trainer_utils
from pathlib import Path
from tqdm import tqdm

from typing import Union, List, Callable

class OpenClipImageEncoderModule(nn.Module):
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

    def remove_text_encoder(self):
        for attr in ["transformer","token_embedding","positional_embedding",
                        "ln_final","text_projection","attn_mask","temp_attn_mask","txt_attn_mask"]:
            if hasattr(self.model, attr):
                delattr(self.model, attr)
        setattr(self.model, "encode_text", None)
        

class OpenClipImageEncoder(BaseModel):
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        mlp_proj:bool = False,
        proj_dim:int = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        self.image_encoder = OpenClipImageEncoderModule(model_name=model_type, pt_weights=pt_weights, keep_lang=False)
        self.feature_dim = self.image_encoder.feature_dim
        self.mlp_proj = mlp_proj
        
        if mlp_proj:
            self.proj_dim = proj_dim
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, proj_dim)
            )
        
    def forward(self, x):
        ftrs = self.image_encoder(x)
        if self.mlp_proj:
            ftrs = self.projector(ftrs)
        return ftrs
    
    @torch.no_grad()
    def predict(self, x):
        """Performs inference (prediction) without gradient computation."""
        preds = self.image_encoder(x)
        return preds
    
    def deactivate_projector(self, remove=False):
        self.mlp_proj = False
        if remove:
            del self.projector
    
    def activate_projector(self, reinitialize=False):
        self.mlp_proj = True
        if reinitialize:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.proj_dim)
            )
    
    def get_train_transforms(self):
        return self.image_encoder.train_preprocess
    
    def get_val_transforms(self):
        return self.image_encoder.val_preprocess
    
    def get_identifier(self):
        return 'Open Clip Image Encoder ' + self.model_type
    
    
    def freeze(self):
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze(self, top_n_blocks=None):
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



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class OpenClipMultiHeadImageClassifier(BaseModel):
    
    def __init__(
        self,
        model_type:str = None,
        pt_weights:str = None,
        datasets_cfgs:dict = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__(loss_fn=loss_fn, metrics=None)
        
        self.model_type = model_type
        self.pt_weights = pt_weights
        self.pretrained = True if pt_weights else False
        
        self.image_encoder = OpenClipImageEncoderModule(model_name=model_type, pt_weights=pt_weights, keep_lang=True)
        
        self.classifier_heads = nn.ModuleDict()
        for dataset_name, class_names in datasets_cfgs.items():
            self.classifier_heads[dataset_name] = _build_classification_head(self.image_encoder.model, dataset_name, class_names, trainer_utils.get_gpu_device())
            
        self.image_encoder.remove_text_encoder()
        # By default, the first head is activated
        self.active_head = next(iter(self.classifier_heads.keys()))
        
        # By default all heads are frozen.
        self.freeze_all_heads()
        
        self.head_metrics = nn.ModuleDict()
        for ds_name, ds_metrics in metrics.items():
            self.head_metrics[ds_name] = nn.ModuleDict()
            for name, metric_instance in ds_metrics.items():
                if not isinstance(metric_instance, torchmetrics.Metric):
                    raise TypeError(f"Metric '{name}' must be an instance of torchmetrics.Metric.")
                self.head_metrics[ds_name][name] = metric_instance
        
        # By default the metric for the defualt active head is activated
        self.metrics = self.head_metrics[self.active_head]
                
                
    def forward(self, x):
        ftrs = self.image_encoder(x)
        logits = self.classifier_heads[self.active_head](ftrs)
        return logits
    
    
    
    def activate_head(self, head_name:str):
        if head_name not in self.classifier_heads:
            raise ValueError('The specified head name is not in the classifier heads.')
        self.active_head = head_name
        self.metrics = self.head_metrics[self.active_head]
        
    def get_active_head(self):
        return self.classifier_heads[self.activate_head]
    
    def get_image_encoder(self):
        return self.image_encoder
    
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
                                      
                        
    
    
def _build_classification_head(
    model: nn.Module, dataset_name: str, class_names: List[str], device: torch.device
) -> ClassificationHead:
    """
    Builds a classification head for a given model and dataset.

    Args:
        model (nn.Module): The model to use for text encoding.
        dataset_name (str): The name of the dataset to use for zero-shot classification.
        class_names (str): List of class names.
        device (torch.device): The device to use for computation.

    Returns:
        A ClassificationHead object with normalized weights for zero-shot classification.
    """
    template = get_clip_templates(dataset_name)

    logit_scale = model.logit_scale
    model.eval()
    model.to(device)

    print(f"Building classification head for {dataset_name}.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(class_names, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    print(f"zeroshot shape, P{zeroshot_weights.shape}")
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    return classification_head