"""
    Code from https://github.com/openai/CLIP/blob/main/clip/model.py
    with SAP baseline implementation
    Also QuickGELU is replaced with nn.GELU
"""

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from copy import deepcopy


from ..sap_utils import forward_cache_activations



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)  # batch_first = False (L, N, E)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()), # Changed from QuickGELU to nn.GELU
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


    @torch.no_grad()
    def get_activations(self, x: torch.Tensor, block_key: str, prev_recur_proj_mat=None):
        """
        Collects 'pre' and 'post' activations for attention (qkv/out_proj) and MLP (linears).
        Mirrors the behavior of your TorchVision EncoderBlock.get_activations but for (L, N, E).
        """
        act = {"pre": OrderedDict(), "post": OrderedDict()}

        # --- Attention (pre-norm)
        x_norm = self.ln_1(x)
        L, N, E = x_norm.shape
        H = self.attn.num_heads
        Hd = E // H

        if prev_recur_proj_mat is None:
            # inputs to qkv projection
            act["pre"][f"{block_key}.self_attn.qkv"] = deepcopy(x_norm.contiguous().view(-1, E).cpu().numpy())

            # q, k, v
            q, k, v = F._in_projection_packed(x_norm, x_norm, x_norm,
                                              self.attn.in_proj_weight, self.attn.in_proj_bias)
            act["post"][f"{block_key}.self_attn.query"] = deepcopy(q.contiguous().view(-1, E).cpu().numpy())
            act["post"][f"{block_key}.self_attn.key"]   = deepcopy(k.contiguous().view(-1, E).cpu().numpy())
            act["post"][f"{block_key}.self_attn.value"] = deepcopy(v.contiguous().view(-1, E).cpu().numpy())

            # reshape for SDPA: (L,N,E) -> (N,H,L,Hd)
            q = q.view(L, N, H, Hd).permute(1, 2, 0, 3)
            k = k.view(L, N, H, Hd).permute(1, 2, 0, 3)
            v = v.view(L, N, H, Hd).permute(1, 2, 0, 3)

            attn_out = F.scaled_dot_product_attention(q, k, v)  # (N,H,L,Hd)
            attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(L * N, E)

            # before out_proj
            act["pre"][f"{block_key}.self_attn.out_proj"] = deepcopy(attn_out.cpu().numpy())

            # out_proj
            attn_out = F.linear(attn_out, self.attn.out_proj.weight, self.attn.out_proj.bias)
            act["post"][f"{block_key}.self_attn.out_proj"] = deepcopy(attn_out.cpu().numpy())

        else:
            # inputs to qkv projection
            x_flat = x_norm.contiguous().view(-1, E)
            act["pre"][f"{block_key}.self_attn.qkv"] = deepcopy(x_flat.cpu().numpy())
            x_flat = torch.matmul(x_flat, prev_recur_proj_mat["pre"][f"{block_key}.self_attn.qkv"])
            x_norm = x_flat.view(L, N, E)

            # q, k, v
            q, k, v = F._in_projection_packed(x_norm, x_norm, x_norm,
                                              self.attn.in_proj_weight, self.attn.in_proj_bias)
            act["post"][f"{block_key}.self_attn.query"] = deepcopy(q.contiguous().view(-1, E).cpu().numpy())
            act["post"][f"{block_key}.self_attn.key"]   = deepcopy(k.contiguous().view(-1, E).cpu().numpy())
            act["post"][f"{block_key}.self_attn.value"] = deepcopy(v.contiguous().view(-1, E).cpu().numpy())

            # apply "pre" projections to q,k,v before SDPA
            q = torch.matmul(q.view(-1, E), prev_recur_proj_mat["pre"][f"{block_key}.self_attn.query"]).view(L, N, E)
            k = torch.matmul(k.view(-1, E), prev_recur_proj_mat["pre"][f"{block_key}.self_attn.key"]).view(L, N, E)
            v = torch.matmul(v.view(-1, E), prev_recur_proj_mat["pre"][f"{block_key}.self_attn.value"]).view(L, N, E)

            q = q.view(L, N, H, Hd).permute(1, 2, 0, 3)
            k = k.view(L, N, H, Hd).permute(1, 2, 0, 3)
            v = v.view(L, N, H, Hd).permute(1, 2, 0, 3)

            attn_out = F.scaled_dot_product_attention(q, k, v)  # (N,H,L,Hd)
            attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(L * N, E)

            # before out_proj
            act["pre"][f"{block_key}.self_attn.out_proj"] = deepcopy(attn_out.cpu().numpy())

            # apply "pre" projection for out_proj input
            attn_out = torch.matmul(attn_out, prev_recur_proj_mat["pre"][f"{block_key}.self_attn.out_proj"])

            # out_proj
            attn_out = F.linear(attn_out, self.attn.out_proj.weight, self.attn.out_proj.bias)
            act["post"][f"{block_key}.self_attn.out_proj"] = deepcopy(attn_out.cpu().numpy())

            # apply "post" projection on out_proj output before residual add
            attn_out = torch.matmul(attn_out, prev_recur_proj_mat["post"][f"{block_key}.self_attn.out_proj"])

        attn_out = attn_out.view(L, N, E)
        x = x + attn_out  # residual after attention

        # --- MLP (pre-norm)
        y = self.ln_2(x)
        mlp_idx = 0
        for layer in self.mlp:
            name = f"{block_key}.mlp.linear{mlp_idx}" if isinstance(layer, nn.Linear) else ""
            if name:
                # flatten to (L*N, feat)
                y_flat = y.contiguous().view(-1, y.shape[-1])
                layer_acts, y_flat = forward_cache_activations(y_flat, layer, name, prev_recur_proj_mat)
                act["pre"].update(layer_acts["pre"])
                act["post"].update(layer_acts["post"])
                y = y_flat.view(L, N, -1)
                mlp_idx += 1
            else:
                y = layer(y)

        x = x + y  # residual after MLP
        return act, x

 
    @torch.no_grad()
    def project_weights(self, projection_mat_dict, block_key: str):
        """
        Projects in/out weights and biases of attention (q,k,v,out_proj) and MLP linears
        using the provided 'pre' and 'post' projection matrices (shape: E×E).
        """
        # -- Attention qkv
        w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3)
        w_q.data = torch.mm(
            projection_mat_dict["post"][f"{block_key}.self_attn.query"].t(),
            torch.mm(w_q.data, projection_mat_dict["pre"][f"{block_key}.self_attn.qkv"].t()),
        )
        w_k.data = torch.mm(
            projection_mat_dict["post"][f"{block_key}.self_attn.key"].t(),
            torch.mm(w_k.data, projection_mat_dict["pre"][f"{block_key}.self_attn.qkv"].t()),
        )
        w_v.data = torch.mm(
            projection_mat_dict["post"][f"{block_key}.self_attn.value"].t(),
            torch.mm(w_v.data, projection_mat_dict["pre"][f"{block_key}.self_attn.qkv"].t()),
        )
        self.attn.in_proj_weight.data = torch.cat((w_q.data, w_k.data, w_v.data), dim=0)

        if self.attn.in_proj_bias is not None:
            b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)
            b_q.data = torch.mm(b_q.data.unsqueeze(0), projection_mat_dict["post"][f"{block_key}.self_attn.query"]).squeeze(0)
            b_k.data = torch.mm(b_k.data.unsqueeze(0), projection_mat_dict["post"][f"{block_key}.self_attn.key"]).squeeze(0)
            b_v.data = torch.mm(b_v.data.unsqueeze(0), projection_mat_dict["post"][f"{block_key}.self_attn.value"]).squeeze(0)
            self.attn.in_proj_bias.data = torch.cat((b_q.data, b_k.data, b_v.data), dim=0)

        # -- Attention out_proj
        self.attn.out_proj.weight.data = torch.mm(
            projection_mat_dict["post"][f"{block_key}.self_attn.out_proj"].t(),
            torch.mm(self.attn.out_proj.weight.data, projection_mat_dict["pre"][f"{block_key}.self_attn.out_proj"].t()),
        )
        if self.attn.out_proj.bias is not None:
            self.attn.out_proj.bias.data = torch.mm(
                self.attn.out_proj.bias.data.unsqueeze(0),
                projection_mat_dict["post"][f"{block_key}.self_attn.out_proj"],
            ).squeeze(0)

        # -- MLP linears
        mlp_idx = 0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                name = f"{block_key}.mlp.linear{mlp_idx}"
                layer.weight.data = torch.mm(
                    projection_mat_dict["post"][name].t(),
                    torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][name].t()),
                ).view_as(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data = torch.mm(layer.bias.data.unsqueeze(0), projection_mat_dict["post"][name]).squeeze(0)
                mlp_idx += 1
        return


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


    @torch.no_grad()
    def get_activations(self, x: torch.Tensor, prev_recur_proj_mat=None):
        """
        Iterate residual blocks and collect activations, returning (act, x_out).
        x is (L, N, E); output is (L, N, E).
        """
        act = {"pre": OrderedDict(), "post": OrderedDict()}
        for i, block in enumerate(self.resblocks):
            block_key = f"encoderblock{i}"
            block_act, x = block.get_activations(x, block_key, prev_recur_proj_mat)
            act["pre"].update(block_act["pre"])
            act["post"].update(block_act["post"])
        return act, x


    @torch.no_grad()
    def project_weights(self, projection_mat_dict):
        for i, block in enumerate(self.resblocks):
            block_key = f"encoderblock{i}"
            block.project_weights(projection_mat_dict, block_key)
        return


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # will be set in get_activations() for parity with TV ViT
        self.max_head_linear_layers = 1

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)                          # [N, C=width, G, G]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [N, width, G^2]
        x = x.permute(0, 2, 1)                     # [N, G^2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)                                  # [N, G^2+1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                     # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                     # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


    @torch.no_grad()
    def get_activations(self, x: torch.Tensor, prev_recur_proj_mat=None):
        """
        Mirrors TorchVision ViT.get_activations():
          - caches conv1, transformer (per-block attention+MLP), and final projection ("heads.linear0")
          - returns {'pre': OrderedDict, 'post': OrderedDict}
        """
        act = {"pre": OrderedDict(), "post": OrderedDict()}
        N, C, H, W = x.shape

        # conv1 (patchify stem)
        layer_acts, x = forward_cache_activations(x, self.conv1, "conv1", prev_recur_proj_mat)
        act["pre"].update(layer_acts["pre"])
        act["post"].update(layer_acts["post"])

        # reshape to tokens
        G2 = x.shape[-2] * x.shape[-1]  # grid^2
        x = x.reshape(N, self.proj.shape[0], -1).permute(0, 2, 1)  # [N, G^2, width]

        # prepend class token, add pos, pre-norm
        cls = self.class_embedding.to(x.dtype)[None, None, :].expand(N, 1, -1)  # [N,1,width]
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # transformer (expects LND)
        x = x.permute(1, 0, 2)
        tr_acts, x = self.transformer.get_activations(x, prev_recur_proj_mat)
        act["pre"].update(tr_acts["pre"])
        act["post"].update(tr_acts["post"])
        x = x.permute(1, 0, 2)

        # take CLS, post-norm
        x = self.ln_post(x[:, 0, :])

        # final projection acts as "heads.linear0" (like TV's heads)
        if self.proj is not None:
            name = "heads.linear0"
            x_flat = x.contiguous().view(-1, x.shape[-1])
            if prev_recur_proj_mat is None:
                act["pre"][name] = deepcopy(x_flat.cpu().numpy())
                out = torch.matmul(x_flat, self.proj)  # (N, width) @ (width, out_dim)
                act["post"][name] = deepcopy(out.cpu().numpy())
            else:
                act["pre"][name] = deepcopy(x_flat.cpu().numpy())
                x_flat = torch.matmul(x_flat, prev_recur_proj_mat["pre"][name])
                out = torch.matmul(x_flat, self.proj)
                act["post"][name] = deepcopy(out.cpu().numpy())
                out = torch.matmul(out, prev_recur_proj_mat["post"][name])
            # we don't need to return logits here; only cache
        self.max_head_linear_layers = 1
        return act


    @torch.no_grad()
    def project_weights(self, projection_mat_dict, proj_classifier: bool = False):
        """
        Projects conv1, per-block transformer weights, and the final projection.
        If proj_classifier=False, mirrors your TV ViT behavior by avoiding left-multiplying the final head.
        """
        # conv1 (treat as linear over im2col)
        self.conv1.weight.data = torch.mm(
            projection_mat_dict["post"]["conv1"].t(),
            torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["pre"]["conv1"].t()),
        ).view_as(self.conv1.weight.data)
        # (bias=False in CLIP conv1)

        # transformer (per-block attention & MLP)
        self.transformer.project_weights(projection_mat_dict)

        # final projection acts like heads.linear0
        name = "heads.linear0"
        if self.proj is not None:
            if not proj_classifier:
                # Equivalent to: W(out,in) @ P_pre^T, but our param is M(in,out) used as x @ M
                # Let W = M^T. W <- W @ P_pre^T  => M <- P_pre @ M
                self.proj.data = torch.mm(projection_mat_dict["pre"][name], self.proj.data)
            else:
                # Full two-sided projection:
                # W' = P_post^T @ W @ P_pre^T  =>  M' = P_pre @ M @ P_post
                self.proj.data = torch.mm(
                    torch.mm(projection_mat_dict["pre"][name], self.proj.data),
                    projection_mat_dict["post"][name],
                )
        return




class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


