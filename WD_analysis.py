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


from helper_funcs import evaluate_model, recalibrate_batchnorm, eval_model_on_clean_noise_splits, search_optimal_coefficient

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
    taskvector1: TaskVector,          # τ_1  (e.g., clean)
    support_tv1: Dataset,             # μ_1
    taskvector2: TaskVector,          # τ_2  (e.g., triggered)
    support_tv2: Dataset,             # μ_2
    calibration_dl: DataLoader,       # For callibrating bn layers for CNNs
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
            if calibration_dl:
                recalibrate_batchnorm(model_tv1, calibration_dl, device)

            model_tv2 = copy.deepcopy(model).to(device)
            taskvector2.apply_to(model_tv2, scaling_coef=float(alpha_tv2), strict=False)
            if calibration_dl:
                recalibrate_batchnorm(model_tv2, calibration_dl, device)

            model_mlt = copy.deepcopy(model).to(device)
            taskvector1.apply_to(model_mlt, scaling_coef=float(alpha_tv1), strict=False)
            taskvector2.apply_to(model_mlt, scaling_coef=float(alpha_tv2), strict=False)
            if calibration_dl:
                recalibrate_batchnorm(model_mlt, calibration_dl, device)

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


def apply_WD_antitask_analysis(
    model: torch.nn.Module,            # θ_pre
    clean_tv: TaskVector,            # τ_c (clean task vector)
    noise_tv: TaskVector,            # τ_n (noise / anti-task vector)
    testset: Dataset,                # same support for both
    alpha_range: tuple,                # (min_alpha, max_alpha)
    step: float,                       # grid step
    batch_size: int,
    device: torch.device,
    metric: str = "loss",              # "loss" (recommended) or "error"
    eps: float = 1e-8
):
    """
    Returns:
      {
        "alphas": np.ndarray,             # 1D grid of alphas
        "risk_base": float,               # R(0,0)
        "risk_c_only": np.ndarray,        # R(α_c,0) [W]
        "risk_n_only": np.ndarray,        # R(0,α_n) [H]
        "risk_map": np.ndarray,           # R(α_c,α_n) [H,W]
        "delta_c": np.ndarray,            # Δ_c(α_c) = R(α_c,0)-R(0,0) [W]
        "delta_n": np.ndarray,            # Δ_n(α_n) = R(0,α_n)-R(0,0) [H]
        "interaction": np.ndarray,        # I(α_c,α_n) [H,W]
        "wd_map": np.ndarray,             # ξ_anti(α_c,α_n) [H,W]
        "best": {"alpha_c": float, "alpha_n": float, "wd": float}
      }
    """
    # -------------------------
    # helpers
    # -------------------------
    def _risk_from_metrics(metrics):
        if metric == "loss":
            # expect metrics["loss"]
            return float(metrics["Loss"])
        elif metric == "error":
            # expect metrics["acc"]
            return float(1.0 - metrics["ACC"])
        else:
            raise ValueError("metric must be 'loss' or 'error'.")

    def _eval_model(m, dl):
        metrics, preds, labels = evaluate_model(m, dl, device)  
        return _risk_from_metrics(metrics)

    # -------------------------
    # prep
    # -------------------------
    alphas = np.arange(alpha_range[0], alpha_range[1] + step, step, dtype=float)
    H = W = len(alphas)

    test_dl = _build_dataloader(testset, batch_size)  # ensure shuffle=False

    risk_map   = np.zeros((H, W), dtype=np.float32)   # R(α_c,α_n)
    interaction = np.zeros_like(risk_map)             # I(α_c,α_n)
    wd_map     = np.zeros_like(risk_map)              # ξ_anti

    # Base risk R(0,0)
    base_model = copy.deepcopy(model).to(device)
    risk_base = _eval_model(base_model, test_dl)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Precompute axis terms: R(α_c,0) and R(0,α_n)
    risk_c_only = np.zeros(W, dtype=np.float32)
    risk_n_only = np.zeros(H, dtype=np.float32)

    # R(α_c,0)
    for xi, a_c in enumerate(tqdm(alphas, desc="Precompute clean axis", leave=False)):
        m_c = copy.deepcopy(model).to(device)
        clean_tv.apply_to(m_c, scaling_coef=float(a_c), strict=False)
        risk_c_only[xi] = _eval_model(m_c, test_dl)
        del m_c
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # R(0,α_n)
    for yi, a_n in enumerate(tqdm(alphas, desc="Precompute noise axis", leave=False)):
        m_n = copy.deepcopy(model).to(device)
        noise_tv.apply_to(m_n, scaling_coef=float(a_n), strict=False)
        risk_n_only[yi] = _eval_model(m_n, test_dl)
        del m_n
        if device.type == "cuda":
            torch.cuda.empty_cache()

    delta_c = risk_c_only - risk_base    # Δ_c(α_c)
    delta_n = risk_n_only - risk_base    # Δ_n(α_n)

    # -------------------------
    # full grid + interaction + normalized WD
    # -------------------------
    pbar = tqdm(total=H*W, desc="Grid (clean, noise)")
    for yi, a_n in enumerate(alphas):            # rows: α_n
        for xi, a_c in enumerate(alphas):        # cols: α_c
            m_cn = copy.deepcopy(model).to(device)
            clean_tv.apply_to(m_cn, scaling_coef=float(a_c), strict=False)
            noise_tv.apply_to(m_cn, scaling_coef=float(a_n), strict=False)
            r_cn = _eval_model(m_cn, test_dl)    # R(α_c,α_n)
            del m_cn
            if device.type == "cuda":
                torch.cuda.empty_cache()

            risk_map[yi, xi] = r_cn

            # I(α_c,α_n) = R(α_c,α_n) - R(α_c,0) - R(0,α_n) + R(0,0)
            I = r_cn - risk_c_only[xi] - risk_n_only[yi] + risk_base
            interaction[yi, xi] = I

            denom = abs(delta_c[xi]) + abs(delta_n[yi]) + eps
            wd_map[yi, xi] = float(abs(I) / denom) if denom > eps else 0.0

            pbar.update(1)
    pbar.close()

    # Best (minimum disentanglement error)
    iy, ix = np.unravel_index(np.argmin(wd_map), wd_map.shape)
    best = {"alpha_c": float(alphas[ix]), "alpha_n": float(alphas[iy]), "wd": float(wd_map[iy, ix])}

    return {
        "alphas": alphas,
        "risk_base": float(risk_base),
        "risk_c_only": risk_c_only,
        "risk_n_only": risk_n_only,
        "risk_map": risk_map,
        "delta_c": delta_c,
        "delta_n": delta_n,
        "interaction": interaction,
        "wd_map": wd_map,
        "best": best,
    }
    

    
@torch.no_grad()
def apply_WD_antitask_analysis_acc(
    model: torch.nn.Module,           # θ_pre
    taskvector1,                      # τ_1 (e.g., "clean")
    taskvector2,                      # τ_2 (e.g., "triggered")
    shared_support,                   # single Dataset S used for both terms
    calibration_dl: DataLoader,
    alpha_range: tuple,               # (min_alpha, max_alpha)
    step: float,                      # grid step
    batch_size: int,
    device: torch.device,
):
    """
    Paper-style Weight Disentanglement on a single shared support S.

    WD at (α1, α2) = 0.5 * [ E_{x∼S} 1( f_{α1,0}(x) != f_{α1,α2}(x) )
                           + E_{x∼S} 1( f_{0,α2}(x) != f_{α1,α2}(x) ) ]

    Returns:
      {
        "alphas": np.ndarray,        # 1D array of alpha values
        "wd_map": np.ndarray,        # [len(alphas), len(alphas)] in [0,1]
        "wd_tv1_only": np.ndarray,   # first term (rows=α2, cols=α1)
        "wd_tv2_only": np.ndarray,   # second term (rows=α2, cols=α1)
        "best": {"alpha_tv1": float, "alpha_tv2": float, "wd": float}
      }
    """
    alphas = np.arange(alpha_range[0], alpha_range[1] + step, step, dtype=float)
    H = W = len(alphas)

    dl = _build_dataloader(shared_support, batch_size)

    # -------------------------------------------
    # Precompute single-axis predictions on S
    # p_tv1[xi, :] = preds of θ + α1 τ1
    # p_tv2[yi, :] = preds of θ + α2 τ2
    # -------------------------------------------
    p_tv1 = []
    for xi, a1 in enumerate(tqdm(alphas, desc="Preds along α1 axis (τ1)", leave=False)):
        m1 = copy.deepcopy(model).to(device)
        taskvector1.apply_to(m1, scaling_coef=float(a1), strict=False)
        if calibration_dl:
            recalibrate_batchnorm(m1, calibration_dl, device)
        _, preds, _ = evaluate_model(m1, dl, device)
        p_tv1.append(preds.view(-1).cpu())
        del m1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    p_tv1 = torch.stack(p_tv1, dim=0)   # [W, N]

    p_tv2 = []
    for yi, a2 in enumerate(tqdm(alphas, desc="Preds along α2 axis (τ2)", leave=False)):
        m2 = copy.deepcopy(model).to(device)
        taskvector2.apply_to(m2, scaling_coef=float(a2), strict=False)
        if calibration_dl:
            recalibrate_batchnorm(m2, calibration_dl, device)
        _, preds, _ = evaluate_model(m2, dl, device)
        p_tv2.append(preds.view(-1).cpu())
        del m2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    p_tv2 = torch.stack(p_tv2, dim=0)   # [H, N]

    # -------------------------------------------
    # Grid over (α1, α2): evaluate combined model once, then compare
    # -------------------------------------------
    wd_map      = np.zeros((H, W), dtype=np.float32)
    wd_tv1_only = np.zeros_like(wd_map)
    wd_tv2_only = np.zeros_like(wd_map)

    N = p_tv1.shape[1]
    pbar = tqdm(total=H * W, desc="Grid (α1, α2)")
    for yi, a2 in enumerate(alphas):          # rows (α2)
        for xi, a1 in enumerate(alphas):      # cols (α1)
            m12 = copy.deepcopy(model).to(device)
            taskvector1.apply_to(m12, scaling_coef=float(a1), strict=False)
            taskvector2.apply_to(m12, scaling_coef=float(a2), strict=False)
            if calibration_dl:
                recalibrate_batchnorm(m12, calibration_dl, device)
            _, p_mlt, _ = evaluate_model(m12, dl, device)
            p_mlt = p_mlt.view(-1).cpu()
            del m12
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Disagreements with shared support S
            d1 = (p_tv1[xi] != p_mlt).float().mean().item() if N else 0.0
            d2 = (p_tv2[yi] != p_mlt).float().mean().item() if N else 0.0

            wd_tv1_only[yi, xi] = d1
            wd_tv2_only[yi, xi] = d2
            wd_map[yi, xi]      = 0.5 * (d1 + d2)

            pbar.update(1)
    pbar.close()

    # Best (minimum WD)
    iy, ix = np.unravel_index(np.argmin(wd_map), wd_map.shape)
    best = {"alpha_tv1": float(alphas[ix]), "alpha_tv2": float(alphas[iy]), "wd": float(wd_map[iy, ix])}

    return {
        "alphas": alphas,
        "wd_map": wd_map,
        "wd_tv1_only": wd_tv1_only,
        "wd_tv2_only": wd_tv2_only,
        "best": best,
    }