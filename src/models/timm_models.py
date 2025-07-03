import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp import autocast

import timm


class TimmModels(nn.Module):
    
    def __init__(
        self,
        model_type:str = None,
        pt_cfg:str = None,
        num_classes:int = None,
        weight_init = None,
        loss_fn:nn.Module = None,
        metrics:dict = None
    ):
        super().__init__()

