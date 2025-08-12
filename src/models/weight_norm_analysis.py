import re
from collections import defaultdict, OrderedDict
from typing import Dict, Iterable, Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Filtering rules ----------

_NORM_HINTS = [
    "batchnorm", "batch_norm", "bn",
    "layernorm", "layer_norm", "ln",
    "groupnorm", "group_norm", "gn",
    "instancenorm", "instance_norm", "inorm", "in"
]

def _looks_like_norm_param(name: str, is_weight_or_bias: bool) -> bool:
    lower = name.lower()
    if not is_weight_or_bias:
        return False
    return any(hint in lower for hint in _NORM_HINTS)

def _is_bias(name: str) -> bool:
    return name.split(".")[-1].lower() == "bias"

def _default_param_selector(name: str) -> bool:
    leaf = name.split(".")[-1].lower()
    is_weight_or_bias = leaf in ("weight", "bias")
    if _is_bias(name):
        return False
    if _looks_like_norm_param(name, is_weight_or_bias):
        return False
    if any(k in leaf for k in ["running_mean", "running_var", "num_batches_tracked"]):
        return False
    return True

# ---------- Norm extraction ----------

def _per_unit_l2_norms(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim >= 2:
        w = weight.reshape(weight.shape[0], -1)
        return torch.linalg.vector_norm(w, ord=2, dim=1)
    elif weight.ndim == 1:
        return weight.abs()
    else:
        return weight.abs().reshape(1)

def _elementwise_l1_norms(weight: torch.Tensor) -> torch.Tensor:
    # element-wise L1 = absolute value for scalars
    return weight.abs().reshape(-1)

def _extract_layer_unit_norms(
    state_dict: Dict[str, torch.Tensor],
    param_selector=_default_param_selector,
    device: Optional[torch.device] = torch.device("cpu")
) -> Dict[str, np.ndarray]:
    out = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not param_selector(name):
            continue
        w = tensor.detach().to(device).float()
        norms = _per_unit_l2_norms(w).cpu().numpy()
        if norms.size > 0:
            out[name] = norms
    return out

def _extract_elementwise_l1_norms(
    state_dict: Dict[str, torch.Tensor],
    param_selector=_default_param_selector,
    device: Optional[torch.device] = torch.device("cpu")
) -> Dict[str, np.ndarray]:
    out = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not param_selector(name):
            continue
        w = tensor.detach().to(device).float()
        norms = _elementwise_l1_norms(w).cpu().numpy()
        if norms.size > 0:
            out[name] = norms
    return out

# ---------- Grouping ----------

def _group_keys_by_prefix(keys: Iterable[str], max_groups: int = 12) -> Dict[str, List[str]]:
    keys = list(keys)
    if len(keys) <= max_groups:
        return {k: [k] for k in keys}

    split_keys = [k.split(".") for k in keys]
    depth = 1
    while True:
        groups = defaultdict(list)
        for k, parts in zip(keys, split_keys):
            prefix = ".".join(parts[:depth]) if len(parts) >= depth else k
            groups[prefix].append(k)
        if len(groups) <= max_groups or depth > max(len(p) for p in split_keys):
            if len(groups) > max_groups:
                items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
                kept = dict(items[:max_groups-1])
                merged_keys = []
                for _, v in items[max_groups-1:]:
                    merged_keys.extend(v)
                kept["(others)"] = merged_keys
                return kept
            return dict(groups)
        depth += 1

# ---------- Density helpers (replace hist bars with curves) ----------

def _density_from_values(values: np.ndarray, bins: int, data_range: Optional[Tuple[float, float]] = None):
    """
    Returns (x_midpoints, density) using numpy.histogram with density=True.
    """
    if values.size == 0:
        return np.array([]), np.array([])
    hist, edges = np.histogram(values, bins=bins, range=data_range, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, hist

def _plot_density(ax, values: np.ndarray, bins: int, label: Optional[str], log: bool, alpha: float = 0.25,
                  data_range: Optional[Tuple[float, float]] = None):
    x, y = _density_from_values(values, bins=bins, data_range=data_range)
    if x.size == 0:
        return
    line, = ax.plot(x, y, label=label)
    ax.fill_between(x, y, step="mid", alpha=alpha)
    if log:
        ax.set_yscale("log")
    # ax.set_xlabel("Value")
    # ax.set_ylabel("Density")

# ---------- Plotting helpers (curves) ----------

def _plot_overall_density(all_values: np.ndarray, bins=50, log=False, title="Density", saving_path=None):
    if all_values.size == 0:
        print("No values to plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_density(ax, all_values, bins=bins, label=None, log=log, alpha=0.25)
    ax.set_title(title)
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path, dpi=300)
    else:
        plt.show()

def _plot_layerwise_grouped_densities(layer_values: Dict[str, np.ndarray],
                                     max_groups=12, bins=40, log=False, suptitle="Layer-wise densities", saving_path=None):
    if not layer_values:
        print("No values to plot.")
        return
    groups = _group_keys_by_prefix(layer_values.keys(), max_groups=max_groups)
    grouped_vals = {
        g: np.concatenate([layer_values[k] for k in keys if k in layer_values])
        for g, keys in groups.items()
    }
    n = len(grouped_vals)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    for ax, (g, arr) in zip(axes.flat, sorted(grouped_vals.items())):
        _plot_density(ax, arr, bins=bins, label=None, log=log, alpha=0.25)
        # ax.set_title(g if len(g) < 40 else g[:37] + "...")
        ax.set_title(g if len(g) < 40 else g[:37] + "...", fontsize=9) 
    # Turn off any leftover axes
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle)
    
    # After plotting all subplots:
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=8)  # smaller tick numbers

    # Shared axis labels closer to plots:
    fig.text(0.5, 0.02, "Value", ha="center", fontsize=10)
    fig.text(0.02, 0.5, "Density", va="center", rotation="vertical", fontsize=10)

    
    fig.tight_layout(pad=2.0, w_pad=0.0, h_pad=3.0)
    if saving_path:
        plt.savefig(saving_path, dpi=300)
    else:
        plt.show()



# -----------------------------------------------------------------------
# ---------- Master functions (single model → now curve-based) ----------

def plot_abs_weight_norms_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    include_bias_and_norm=False,
    max_groups=12,
    overall_bins=50,
    layer_bins=40,
    logy=False,
    saving_path: Path = None
):
    selector = (lambda n: True) if include_bias_and_norm else _default_param_selector

    elem_l1 = _extract_elementwise_l1_norms(state_dict, param_selector=selector)
    if not elem_l1:
        print("No parameters selected.")
        return
    _plot_overall_density(np.concatenate(list(elem_l1.values())), bins=overall_bins, log=logy,
                         title="Whole-model element-wise L1 norms (abs weights)")
    _plot_layerwise_grouped_densities(elem_l1, max_groups=max_groups, bins=layer_bins, log=logy,
                                     suptitle="Layer-wise element-wise L1 norms (grouped)")

def plot_l2_weight_norms_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    include_bias_and_norm=False,
    max_groups=12,
    overall_bins=50,
    layer_bins=40,
    logy=False,
    saving_path: Path = None
):
    selector = (lambda n: True) if include_bias_and_norm else _default_param_selector

    unit_l2 = _extract_layer_unit_norms(state_dict, param_selector=selector)
    if not unit_l2:
        print("No parameters selected.")
        return
    _plot_overall_density(np.concatenate(list(unit_l2.values())), bins=overall_bins, log=logy,
                         title="Whole-model per-unit L2 norms")
    _plot_layerwise_grouped_densities(unit_l2, max_groups=max_groups, bins=layer_bins, log=logy,
                                     suptitle="Layer-wise per-unit L2 norms (grouped)")



# -----------------------------------------------------------------------
# ---------- Multi-model comparison ----------

def _collect_all_values_across_models(layer_values_by_model: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    all_vals = []
    for d in layer_values_by_model.values():
        for arr in d.values():
            if arr.size > 0:
                all_vals.append(arr)
    if not all_vals:
        return np.array([])
    return np.concatenate(all_vals)

def _plot_overall_compare(layer_values_by_model: Dict[str, Dict[str, np.ndarray]],
                          bins=50, log=False, title="Overall comparison", saving_path=None):
    pooled = _collect_all_values_across_models(layer_values_by_model)
    if pooled.size == 0:
        print("No values to plot.")
        return
    data_range = (pooled.min(), pooled.max())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, layer_vals in layer_values_by_model.items():
        vals = np.concatenate([v for v in layer_vals.values()]) if layer_vals else np.array([])
        if vals.size == 0:
            continue
        _plot_density(ax, vals, bins=bins, label=label, log=log, alpha=0.25, data_range=data_range)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path, dpi=300)
    else:
        plt.show()

def _plot_layerwise_grouped_compare(layer_values_by_model: Dict[str, Dict[str, np.ndarray]],
                                    max_groups=12, bins=40, log=False, suptitle="Layer-wise comparison", saving_path=None):
    # Derive groups from the union of keys across models
    all_keys = set()
    for d in layer_values_by_model.values():
        all_keys |= set(d.keys())
    if not all_keys:
        print("No values to plot.")
        return
    groups = _group_keys_by_prefix(sorted(all_keys), max_groups=max_groups)

    # Precompute pooled ranges per group for shared bins across models
    grouped_pooled_ranges = OrderedDict()
    for g, keys in groups.items():
        pooled = []
        for d in layer_values_by_model.values():
            pooled.extend([d[k] for k in keys if k in d and d[k].size > 0])
        if pooled:
            pooled = np.concatenate(pooled)
            grouped_pooled_ranges[g] = (pooled.min(), pooled.max())
        else:
            grouped_pooled_ranges[g] = None

    n = len(groups)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)

    for ax, (g, keys) in zip(axes.flat, sorted(groups.items())):
        data_range = grouped_pooled_ranges[g]
        has_any = False
        for label, d in layer_values_by_model.items():
            arrs = [d[k] for k in keys if k in d and d[k].size > 0]
            if not arrs:
                continue
            vals = np.concatenate(arrs)
            _plot_density(ax, vals, bins=bins, label=label, log=log, alpha=0.25, data_range=data_range)
            has_any = True
        # ax.set_title(g if len(g) < 40 else g[:37] + "...")
        ax.set_title(g if len(g) < 40 else g[:37] + "...", fontsize=9)
        if has_any:
            ax.legend(fontsize="small")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])

    # turn off extras
    for ax in axes.flat[n:]:
        ax.axis("off")
    
        
    # After plotting all subplots:
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=8)  # smaller tick numbers

    # Shared axis labels closer to plots:
    fig.text(0.5, 0.01, "Value", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "Density", va="center", rotation="vertical", fontsize=12)


    fig.suptitle(suptitle)
    fig.tight_layout(pad=2.0, w_pad=0.0, h_pad=3.0)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
    if saving_path:
        plt.savefig(saving_path, dpi=300)
    else:
        plt.show()

def plot_abs_weight_norms_compare(
    state_dicts: Dict[str, Dict[str, torch.Tensor]],
    *,
    include_bias_and_norm: bool = False,
    max_groups: int = 40,
    overall_bins: int = 200,
    layer_bins: int = 200,
    logy: bool = False,
    device: Optional[torch.device] = torch.device("cpu"),
    saving_path: Path = None
):
    """
    Overlay element-wise L1 norm densities for multiple checkpoints of the same model.
    state_dicts: mapping from label -> state_dict
    """
    selector = (lambda n: True) if include_bias_and_norm else _default_param_selector

    # Extract per-model
    per_model_elem_l1: Dict[str, Dict[str, np.ndarray]] = OrderedDict()
    for label, sd in state_dicts.items():
        per_model_elem_l1[label] = _extract_elementwise_l1_norms(sd, param_selector=selector, device=device)
        
    
    path_lw = saving_path.with_name(f"{saving_path.stem}_lw{saving_path.suffix}")
    path_oa = saving_path.with_name(f"{saving_path.stem}_oa{saving_path.suffix}")

    _plot_overall_compare(per_model_elem_l1, bins=overall_bins, log=logy,
                          title="Whole-model element-wise L1 norms (comparison)", saving_path=path_oa)
    _plot_layerwise_grouped_compare(per_model_elem_l1, max_groups=max_groups, bins=layer_bins, log=logy,
                                    suptitle="Layer-wise element-wise L1 norms (grouped, comparison)", saving_path=path_lw)

def plot_l2_weight_norms_compare(
    state_dicts: Dict[str, Dict[str, torch.Tensor]],
    *,
    include_bias_and_norm: bool = False,
    max_groups: int = 12,
    overall_bins: int = 60,
    layer_bins: int = 50,
    logy: bool = False,
    device: Optional[torch.device] = torch.device("cpu"),
    saving_path: Path = None
):
    """
    Overlay per-unit L2 norm densities for multiple checkpoints of the same model.
    state_dicts: mapping from label -> state_dict
    """
    selector = (lambda n: True) if include_bias_and_norm else _default_param_selector

    # Extract per-model
    per_model_unit_l2: Dict[str, Dict[str, np.ndarray]] = OrderedDict()
    for label, sd in state_dicts.items():
        per_model_unit_l2[label] = _extract_layer_unit_norms(sd, param_selector=selector, device=device)
        
    path_lw = saving_path.with_name(f"{saving_path.stem}_lw{saving_path.suffix}")
    path_oa = saving_path.with_name(f"{saving_path.stem}_oa{saving_path.suffix}")

    _plot_overall_compare(per_model_unit_l2, bins=overall_bins, log=logy,
                          title="Whole-model per-unit L2 norms (comparison)", saving_path=path_oa)
    _plot_layerwise_grouped_compare(per_model_unit_l2, max_groups=max_groups, bins=layer_bins, log=logy,
                                    suptitle="Layer-wise per-unit L2 norms (grouped, comparison)", saving_path=path_lw)
