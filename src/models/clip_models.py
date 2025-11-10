
import torch
import torch.nn as nn
from typing import Union, List, Dict
from tqdm import tqdm


from .openai_clip import clip


from ..datasets import get_clip_templates
from ..trainers import utils as trainer_utils
from . import BaseModel  


class CLIPImageEncoderModule(nn.Module):


    def __init__(self, model_name: str, keep_lang: bool = False, device: torch.device = None):
        super().__init__()

        
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = clip.load(model_name, device="cpu")  # construct on cpu initially
        

        # compute feature dim same as before
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
            self.remove_text_encoder()

        # move model to desired device if provided
        if device is not None:
            self.model.to(device)

    def forward(self, images):
        return self.model.encode_image(images)
    
    
    # ------------------------------
    # SAP related methods
    def get_activations(self, x: torch.Tensor, prev_recur_proj_mat: dict = None):
        return self.model.visual.get_activations(x, prev_recur_proj_mat)
    
    def project_weights(self, projection_mat_dict: dict):
        return self.model.visual.project_weights(projection_mat_dict)
    # ------------------------------
    

    def remove_text_encoder(self):
        # remove or nullify text-related attributes (similar to your open_clip removal)
        for attr in ["transformer", "token_embedding", "positional_embedding",
                     "ln_final", "text_projection", "attn_mask", "temp_attn_mask", "txt_attn_mask"]:
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except Exception:
                    # if attribute isn't removable (rare), ignore
                    pass
        # neutralize encode_text function
        setattr(self.model, "encode_text", None)





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


class CLIPMultiHeadImageClassifier(BaseModel):
    def __init__(self, model_type: str = None, datasets_cfgs: dict = None,
                 loss_fn: nn.Module = None, metrics: dict = None):
        super().__init__(loss_fn=loss_fn, metrics=None)

        self.model_type = model_type
    
        self.image_encoder = CLIPImageEncoderModule(model_name=model_type, keep_lang=True, device=trainer_utils.get_gpu_device())

        self.classifier_heads = nn.ModuleDict()
        for dataset_name, class_names in datasets_cfgs.items():
            self.classifier_heads[dataset_name] = self._build_classification_head(dataset_name, class_names, device=trainer_utils.get_gpu_device())

        # remove text encoder after building heads just like before
        self.image_encoder.remove_text_encoder()
        # default active head
        self.active_head = next(iter(self.classifier_heads.keys()))

        # freeze heads by default
        self.freeze_all_heads()

        self.head_metrics = nn.ModuleDict()
        for ds_name, ds_metrics in metrics.items():
            self.head_metrics[ds_name] = nn.ModuleDict()
            for name, metric_instance in ds_metrics.items():
                if not isinstance(metric_instance, torch.nn.Module):  # torchmetrics.Metric inherits Module
                    raise TypeError(f"Metric '{name}' must be an instance of torchmetrics.Metric.")
                self.head_metrics[ds_name][name] = metric_instance

        self.metrics = self.head_metrics[self.active_head]

    def forward(self, x):
        ftrs = self.image_encoder(x)
        logits = self.classifier_heads[self.active_head](ftrs)
        return logits
    
    # ------------------------------
    # SAP related methods
    def get_activations(self, x: torch.Tensor, prev_recur_proj_mat: dict = None):
        return self.image_encoder.get_activations(x, prev_recur_proj_mat)
    
    def project_weights(self, projection_mat_dict: dict, project_classifier_head: bool = False):
        # For CLIP we neglect the `project_classifier_head` since CLIP classification head is frozen and static.
        return self.image_encoder.project_weights(projection_mat_dict)

    # ------------------------------
    def activate_head(self, head_name: str):
        if head_name not in self.classifier_heads:
            raise ValueError('The specified head name is not in the classifier heads.')
        self.active_head = head_name
        self.metrics = self.head_metrics[self.active_head]
    

    def get_active_head(self):
        return self.classifier_heads[self.active_head]

    def get_feature_extractor(self):
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


    def freeze_head(self, head_name: Union[str, List[str]]):
        if isinstance(head_name, list):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(False)
        elif isinstance(head_name, str):
            self.classifier_heads[head_name].requires_grad_(False)

    def freeze_all_heads(self):
        for _, cls_head in self.classifier_heads.items():
            cls_head.requires_grad_(False)

    def unfreeze_head(self, head_name: Union[str, List[str]]):
        if isinstance(head_name, list):
            for hn in head_name:
                self.classifier_heads[hn].requires_grad_(True)
        elif isinstance(head_name, str):
            self.classifier_heads[head_name].requires_grad_(True)

    def unfreeze_all_heads(self):
        for _, cls_head in self.classifier_heads.items():
            cls_head.requires_grad_(True)

    def freeze_encoder(self):
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self, top_n_blocks=None):
        # match your previous behavior (works for ViTs with .visual.transformer.resblocks)
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
        for attr in ["ln_post", "proj", "ln_pre"]:
            mod = getattr(self.image_encoder.model.visual, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)

    def get_identifier(self):
        return 'OpenAI Clip Image Classifier ' + str(self.model_type)

    def _build_classification_head(self, dataset_name: str, class_names: List[str], device: torch.device):
        """
        Build the same zero-shot classification head using OpenAI clip model APIs.
        """
        template = get_clip_templates(dataset_name)
        model = self.image_encoder.model

        logit_scale = model.logit_scale
        model.to(device)
        model.eval()

        
        print(f"Building classification head for {dataset_name}.")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(class_names, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                texts = []
                for t in template:
                    texts.append(t(classname))
                texts = clip.tokenize(texts).to(device)  # tokenize
                
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
        # print(zeroshot_weights.mean(), zeroshot_weights.std())
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
        return classification_head
