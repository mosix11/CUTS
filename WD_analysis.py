import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from src.datasets import dataset_factory, dataset_wrappers, BaseClassificationDataset
from src.models import model_factory, TaskVector
from src.utils import nn_utils, misc_utils
import copy
from tqdm import tqdm
from torchmetrics.classification import MulticlassConfusionMatrix
from typing import Union, Tuple, List


from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient

def _build_dataloader(dataset, batch_size):
    return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )


def apply_WD_analysis(
    model: torch.nn.Module,           # θ_pre (initialization model)
    taskvector1: "TaskVector",        # τ_1  (e.g., clean)
    support_tv1: "Dataset",           # μ_1
    taskvector2: "TaskVector",        # τ_2  (e.g., triggered)
    support_tv2: "Dataset",           # μ_2
    alhpa_range: tuple,               # (min_alpha, max_alpha)
    step: float,                      # grid step
    batch_size: int,
    device: torch.device
):
    """
    Returns:
      {
        "alphas": np.ndarray,        # 1D array of alpha values
        "wd_map": np.ndarray,        # [len(alphas_2), len(alphas_1)] in [0,1]
        "wd_tv1_only": np.ndarray,   # E_{x~μ1}[1( f(x;θ+α1τ1) != f(x;θ+α1τ1+α2τ2) )]
        "wd_tv2_only": np.ndarray,   # E_{x~μ2}[1( f(x;θ+α2τ2) != f(x;θ+α1τ1+α2τ2) )]
        "best": {"alpha_tv1": float, "alpha_tv2": float, "wd": float}
      }
    """
    # alpha axis
    alphas = np.arange(alhpa_range[0], alhpa_range[1] + step, step, dtype=float)

    # build loaders (ensure shuffle=False in _build_dataloader)
    support_tv1_dl = _build_dataloader(support_tv1, batch_size)
    support_tv2_dl = _build_dataloader(support_tv2, batch_size)

    H, W = len(alphas), len(alphas)  # rows: alpha_tv2 (y), cols: alpha_tv1 (x)
    wd_map      = np.zeros((H, W), dtype=np.float32)
    wd_tv1_only = np.zeros_like(wd_map)  # μ1 contribution
    wd_tv2_only = np.zeros_like(wd_map)  # μ2 contribution

    pbar = tqdm(total=H * W, desc="Applying alpha combinations", leave=True)

    for yi, alpha_tv2 in enumerate(alphas):          # y-axis: τ2 scale
        for xi, alpha_tv1 in enumerate(alphas):      # x-axis: τ1 scale
            # Build three models: θ+α1τ1, θ+α2τ2, θ+α1τ1+α2τ2
            model_tv1 = copy.deepcopy(model).to(device)
            taskvector1.apply_to(model_tv1, scaling_coef=float(alpha_tv1), strict=False)

            model_tv2 = copy.deepcopy(model).to(device)
            taskvector2.apply_to(model_tv2, scaling_coef=float(alpha_tv2), strict=False)

            model_mlt = copy.deepcopy(model).to(device)
            taskvector1.apply_to(model_mlt, scaling_coef=float(alpha_tv1), strict=False)
            taskvector2.apply_to(model_mlt, scaling_coef=float(alpha_tv2), strict=False)

            # Evaluate on each task's own support (your function returns: metrics, preds, labels)
            _, preds_tv1_on_s1, _ = evaluate_model(model_tv1, support_tv1_dl, device)
            _, preds_mlt_on_s1, _ = evaluate_model(model_mlt, support_tv1_dl, device)

            _, preds_tv2_on_s2, _ = evaluate_model(model_tv2, support_tv2_dl, device)
            _, preds_mlt_on_s2, _ = evaluate_model(model_mlt, support_tv2_dl, device)

            # Ensure tensors and same device (we'll compute on CPU safely)
            p1  = preds_tv1_on_s1.view(-1).cpu()
            pm1 = preds_mlt_on_s1.view(-1).cpu()
            p2  = preds_tv2_on_s2.view(-1).cpu()
            pm2 = preds_mlt_on_s2.view(-1).cpu()

            # Disagreement rates (0/1), then average the two
            err_μ1 = (p1 != pm1).float().mean().item() if p1.numel() else 0.0
            err_μ2 = (p2 != pm2).float().mean().item() if p2.numel() else 0.0

            wd_tv1_only[yi, xi] = err_μ1
            wd_tv2_only[yi, xi] = err_μ2
            wd_map[yi, xi]      = 0.5 * (err_μ1 + err_μ2)

            # cleanup
            del model_tv1, model_tv2, model_mlt
            if device.type == "cuda":
                torch.cuda.empty_cache()

            pbar.update(1)

    pbar.close()

    # Find minimum WD point
    best_idx = np.unravel_index(np.argmin(wd_map), wd_map.shape)
    best = {
        "alpha_tv1": float(alphas[best_idx[1]]),   # column index → α for taskvector1
        "alpha_tv2": float(alphas[best_idx[0]]),   # row index → α for taskvector2
        "wd": float(wd_map[best_idx]),
    }

    return {
        "alphas": alphas,
        "wd_map": wd_map,              # multiply by 100 for percent if plotting with a % colorbar
        "wd_tv1_only": wd_tv1_only,
        "wd_tv2_only": wd_tv2_only,
        "best": best,
    }


# def apply_WD_analysis(
#     model:torch.nn.Module, # This is the initialization model (before training on mix)
#     taskvector1: TaskVector,
#     support_tv1: Dataset,
#     taskvector2: TaskVector,
#     support_tv2: Dataset,
#     alhpa_range: Tuple[float, float],
#     step: int,
#     batch_size: int,
#     device:torch.device
# ):
        
#     alphas = np.arange(alhpa_range[0], alhpa_range[1] + step, step, dtype=float)

#     support_tv1_dl = _build_dataloader(support_tv1, batch_size)  
#     support_tv2_dl = _build_dataloader(support_tv2, batch_size)

#     H, W = len(alphas), len(alphas)  # rows: alpha_tv2, cols: alpha_tv1
#     wd_map   = np.zeros((H, W), dtype=np.float32)
#     wd_tv1_only = np.zeros_like(wd_map)  # μ_c contribution
#     wd_tv2_only = np.zeros_like(wd_map)  # μ_t contribution

#     total_iters = H * W
#     pbar = tqdm(total=total_iters, desc="Applying alpha combinations", leave=True)

#     for yi, alpha_tv2 in enumerate(alphas):      # rows (y-axis)
#         for xi, alpha_tv1 in enumerate(alphas):  # cols (x-axis)
            
#             model_tv1 = copy.deepcopy(model)
#             taskvector1.apply_to(model_tv1, scaling_coef=alpha_tv1, strict=False)
#             model_tv2 = copy.deepcopy(model)
#             taskvector2.apply_to(model_tv2, scaling_coef=alpha_tv2, strict=False)
#             model_mlt = copy.deepcopy(model)
#             taskvector1.apply_to(model_mlt, scaling_coef=alpha_tv1, strict=False)
#             taskvector2.apply_to(model_mlt, scaling_coef=alpha_tv2, strict=False)
            
            
#             _, model_tv1_preds, _ = evaluate_model(model_tv1, support_tv1_dl, device)
#             _, model_mtl_s1_preds, _ = evaluate_model(model_mlt, support_tv1_dl, device)
#             _, model_tv2_preds, _ = evaluate_model(model_tv2, support_tv2_dl, device)
#             _, model_mtl_s2_preds, _ = evaluate_model(model_mlt, support_tv2_dl, device)
            
            

#             # Disagreement on each task's own support
#             err_c = _disagreement_rate(support_tv1_dl, m_c, m_ct, device)  # E_{x~μ_c}[1(…)]
#             err_t = _disagreement_rate(support_tv2_dl, m_t, m_ct, device)  # E_{x~μ_t}[1(…)]

#             wd_c_only[yi, xi] = err_c
#             wd_t_only[yi, xi] = err_t
#             wd_map[yi, xi]    = 0.5 * (err_c + err_t)   # average → in [0,1]

#             pbar.update(1)

#             # Free some memory (important if models are large)
#             del m_c, m_t, m_ct
#             torch.cuda.empty_cache() if device.type == "cuda" else None

#     pbar.close()

#     # Find best (minimum WD)
#     best_idx = np.unravel_index(np.argmin(wd_map), wd_map.shape)
#     best = {
#         "alpha_tv1": float(alphas[best_idx[1]]),  # column index → α_tv1
#         "alpha_tv2": float(alphas[best_idx[0]]),  # row index → α_tv2
#         "wd": float(wd_map[best_idx])
#     }

#     return {
#         "alphas": alphas,
#         "wd_map": wd_map,           # multiply by 100 for percent if desired
#         "wd_c_only": wd_c_only,
#         "wd_t_only": wd_t_only,
#         "best": best,
#     }
        