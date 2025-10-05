import copy
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict

# -----------------------------
# Dataloader helper 
# -----------------------------

def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def _build_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

# -----------------------------
# Small numeric helpers
# -----------------------------
def _center(vec: torch.Tensor) -> torch.Tensor:
    return vec - vec.mean()

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = _center(a.flatten().float())
    b = _center(b.flatten().float())
    denom = (a.norm() * b.norm()).clamp_min(eps)
    if denom.item() == 0.0:
        return 0.0
    return float((a @ b) / denom)

# -----------------------------
# Scalar score definitions
# -----------------------------
def _margin_to_true(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    s_clean(x,y) = z_y - max_{k != y} z_k
    """
    N, K = logits.shape
    arange = torch.arange(N, device=logits.device)
    zy = logits[arange, y]
    masked = logits.clone()
    masked[arange, y] = float("-inf")
    max_other = masked.max(dim=1).values
    return zy - max_other

def _margin_to_target(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    """
    Poison score: s_poison(x) = z_t - max_{k != t} z_k
    """
    N, K = logits.shape
    zt = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = float("-inf")
    max_other = masked.max(dim=1).values
    return zt - max_other

def _symmetric_noise_score(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    s_corr^sym(x,y) = mean_{k != y} z_k - z_y
    (Scale factors like eta/(K-1) cancel in cosine after centering.)
    """
    N, K = logits.shape
    arange = torch.arange(N, device=logits.device)
    zy = logits[arange, y]
    mean_wrong = (logits.sum(dim=1) - zy) / (K - 1)
    return mean_wrong - zy

def _asymmetric_noise_score(logits: torch.Tensor, y: torch.Tensor, noise_map: Dict[int, int]) -> torch.Tensor:
    """
    s_corr^asym(x,y) = z_{kappa(y)} - z_y
    """
    N, K = logits.shape
    device = logits.device
    # Build vectorized map tensor of shape [K]
    map_list = list(range(K))
    for src, dst in noise_map.items():
        if 0 <= src < K and 0 <= dst < K:
            map_list[src] = dst
    map_tensor = torch.tensor(map_list, dtype=torch.long, device=device)
    y_map = map_tensor[y]
    arange = torch.arange(N, device=device)
    return logits[arange, y_map] - logits[arange, y]

# -----------------------------
# Score streaming from a model
# -----------------------------
@torch.no_grad()
def _stream_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str,
    *,
    poison_target_class: int = 0,
    noise_map: Optional[Dict[int, int]] = None,
):
    """
    mode ∈ {'clean', 'poison', 'label_noise_sym', 'label_noise_asym'}
    Returns a single concatenated 1-D tensor of scalar scores for all samples in dataloader.
    Assumes `prepare_batch` is defined in your codebase (same as in evaluate_model).
    """
    model.to(device)
    model.eval()
    scores = []

    for batch in dataloader:
        batch = prepare_batch(batch, device)
        x, y = batch[:2]  # y needed for clean/label-noise

        # reuse your validation_step to obtain logits
        _, preds = model.validation_step(x, y, use_amp=False, return_preds=True)
        logits = preds  # [B, K]

        if mode == 'clean':
            s = _margin_to_true(logits, y)
        elif mode == 'poison':
            s = _margin_to_target(logits, poison_target_class)
        elif mode == 'label_noise_asym':
            if noise_map is None:
                raise ValueError("label_noise_asym mode requires a noise_map.")
            s = _asymmetric_noise_score(logits, y, noise_map)
        elif mode == 'label_noise_sym':
            s = _symmetric_noise_score(logits, y)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")
        scores.append(s.detach().cpu())

    return torch.cat(scores, dim=0)

# -----------------------------
# Hard-coded asymmetric maps
# -----------------------------
def _normalize_dataset_name(name: str) -> str:
    """
    Normalizes dataset name to one of: {'MNIST','CIFAR10','CIFAR100'}
    """
    name = name.strip().upper().replace(" ", "").replace("_", "").replace("-", "")
    if name in {"MNIST"}:
        return "MNIST"
    if name in {"CIFAR10"}:
        return "CIFAR10"
    if name in {"CIFAR100"}:
        return "CIFAR100"
    raise ValueError(f"Unsupported dataset_name '{name}'. Expected MNIST, CIFAR-10, or CIFAR-100.")

def _get_asymmetric_noise_map(dataset_name: str) -> Dict[int, int]:
    """
    Returns the dataset-specific asymmetric noise mapping you specified.
    """
    name = _normalize_dataset_name(dataset_name)

    if name == "MNIST":
        # {7→1, 2→7, 5↔6, 3→8}
        return {7: 1, 2: 7, 5: 6, 6: 5, 3: 8}

    if name == "CIFAR10":
        # {9→1, 2→0, 3↔5, 4→7}
        return {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}

    if name == "CIFAR100":
        # Build cycle-within-superclass mapping from the provided coarse labels
        coarse_labels = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ]
        super_class_map = [[] for _ in range(20)]
        for i, c in enumerate(coarse_labels):
            super_class_map[c].append(i)

        noise_map: Dict[int, int] = {}
        for group in super_class_map:
            L = len(group)
            for idx in range(L):
                src = group[idx]
                dst = group[(idx + 1) % L]
                noise_map[src] = dst
        return noise_map

    # Should never reach here due to normalization guard
    raise ValueError(f"Unsupported dataset_name '{dataset_name}'.")

# -----------------------------
# Main API (updated signature)
# -----------------------------
def compute_task_vector_alignment(
    model: torch.nn.Module,            # θ_pre
    clean_tv,                          # τ_c (TaskVector)
    corruption_tv,                     # τ_n (TaskVector)
    testset_tv1,                       # Dataset (required)
    testset_tv2,                       # Dataset or None
    batch_size: int,
    device: torch.device,
    dataset_name: str,                 # 'MNIST' | 'CIFAR-10' | 'CIFAR-100'
    corruption_type: str,              # 'sym' | 'asym' | 'pois'
    poison_target_class: int = 0,      # poison target (you said it's always 0)
) -> float:
    """
    Returns the alignment score between the clean and corruption task vectors.

    - corruption_type == 'pois':
        * If both testset_tv1 and testset_tv2 are provided:
            Paper's two-set alignment (average of cosines on D1 clean-margins + D2 poison-margins).
        * If only testset_tv1 is provided:
            Single-set poison alignment (both models scored with poison-margin on D1).

    - corruption_type in {'sym', 'asym'}:
        Single-set alignment on testset_tv1:
            cosine( s_clean(clean-TV), s_corr(corruption-TV) ).
        For 'asym', uses your hard-coded dataset-specific noise map.
    """
    # Normalize inputs
    corr_type = corruption_type.strip().lower()
    if corr_type not in {"sym", "asym", "pois"}:
        raise ValueError(f"corruption_type must be one of 'sym', 'asym', 'pois'; got '{corruption_type}'.")

    # Make two edited models WITHOUT mutating the original
    model_clean = copy.deepcopy(model)
    model_corr  = copy.deepcopy(model)

    # Apply task vectors (scaling 1.0 recommended to reflect the learned deltas)
    clean_tv.apply_to(model_clean, scaling_coef=1.0, strict=False)
    corruption_tv.apply_to(model_corr,  scaling_coef=1.0, strict=False)

    dl1 = _build_dataloader(testset_tv1, batch_size)

    # ---------------- Poison triggers ----------------
    if corr_type == "pois":
        # Two-set (paper-style) if tv2 provided
        if testset_tv2 is not None:
            dl2 = _build_dataloader(testset_tv2, batch_size)

            # D1: clean-margins (same scalar def for both models)
            s_clean_D1 = _stream_scores(model_clean, dl1, device, mode='clean')
            s_corr_D1  = _stream_scores(model_corr,  dl1, device, mode='clean')
            cos1 = _cosine_similarity(s_clean_D1, s_corr_D1)

            # D2: poison-margins (same scalar def for both models)
            s_clean_D2 = _stream_scores(model_clean, dl2, device, mode='poison', poison_target_class=poison_target_class)
            s_corr_D2  = _stream_scores(model_corr,  dl2, device, mode='poison', poison_target_class=poison_target_class)
            cos2 = _cosine_similarity(s_clean_D2, s_corr_D2)

            return 0.5 * (cos1 + cos2)

        # Single-set poison alignment (falls back gracefully)
        s_clean = _stream_scores(model_clean, dl1, device, mode='poison', poison_target_class=poison_target_class)
        s_corr  = _stream_scores(model_corr,  dl1, device, mode='poison', poison_target_class=poison_target_class)
        return _cosine_similarity(s_clean, s_corr)

    # --------------- Label noise (sym / asym) ---------------
    if corr_type == "asym":
        noise_map = _get_asymmetric_noise_map(dataset_name)
        label_noise_mode = 'label_noise_asym'
    else:
        noise_map = None
        label_noise_mode = 'label_noise_sym'

    s_clean = _stream_scores(model_clean, dl1, device, mode='clean')
    s_corr  = _stream_scores(
        model_corr, dl1, device,
        mode=label_noise_mode,
        noise_map=noise_map
    )

    return _cosine_similarity(s_clean, s_corr)
