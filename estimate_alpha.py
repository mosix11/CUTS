import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import deque
import math
from sklearn.neighbors import NearestNeighbors
# -----------------------------
# Batch prep + feature extraction
# -----------------------------
def prepare_batch(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]

@torch.no_grad()
def extract_features(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    normalize: bool = True,
    device: torch.device = torch.device('cpu'),
):
    """
    Returns:
        feats: (N, D) float32 tensor on CPU
        labels: (N,) long tensor on CPU  (unused downstream; kept for symmetry)
    """
    feature_extractor.to(device).eval()
    feats, labels = [], []
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        x, y = prepare_batch(batch, device)[:2]
        z = feature_extractor(x)
        # handle transformer-like outputs
        if not isinstance(z, torch.Tensor):
            z2 = getattr(z, "pooler_output", None)
            if z2 is None:
                z2 = z.last_hidden_state[:, 0, :]
            z = z2
        if normalize:
            z = F.normalize(z, dim=1)
        feats.append(z.detach().cpu())
        labels.append(y.detach().cpu().long())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

@torch.no_grad()
def logits_from_features(classifier: nn.Module, feats: torch.Tensor, bs: int = 4096, device: str = "cuda"):
    """
    Args:
        feats: (N, D) on CPU
    Returns:
        logits: (N, K) on CPU (float32)
    """
    classifier = classifier.to(device).eval()
    outs = []
    N = feats.size(0)
    for i in range(0, N, bs):
        fb = feats[i:i+bs].to(device, non_blocking=True)
        outs.append(classifier(fb).detach().float().cpu())
    return torch.cat(outs, dim=0)

# -----------------------------
# Unsupervised score (diversity-aware)
# -----------------------------
def _knn_indices(feats_np: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k + 1, len(feats_np))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm='auto')
    nbrs.fit(feats_np)
    _, idx = nbrs.kneighbors(feats_np, return_distance=True)
    return idx[:, 1:]  # drop self

def knn_self_agreement_diversity(
    feats: torch.Tensor,
    probs: torch.Tensor,
    *,
    k: int,
    num_clusters: int,
    coverage_rate: float = 1.0,          # e.g., 0.6–0.8 for CIFAR-100 with ~2k samples
    cov_penalty_weight: float = 1.0,     # penalty for coverage shortfall
    hard_min_per_class: int | None = None,
    random_state: int = 0,
) -> float:
    """
    Chance-corrected kNN self-agreement + coverage-aware diversity constraints.

    - Requires only ceil(coverage_rate * num_clusters) classes to have >= hard_min_per_class support.
    - Penalizes (i) effective class shortfall vs required and (ii) coverage shortfall.
    - Optional NMI bonus aligns predictions with k-means(num_clusters=required) clusters.
    """
    eps = 1e-8
    N, K = probs.shape
    feats_np = feats.numpy()
    probs_np = probs.numpy()
    yhat = probs_np.argmax(axis=1)

    # --- class support counts
    counts = np.bincount(yhat, minlength=K)
    if hard_min_per_class is None:
        hard_min_per_class = min(k + 1, max(3, k))
    num_supported = int((counts >= hard_min_per_class).sum())

    # coverage requirement (SOFT)
    required = max(2, min(K, math.ceil(coverage_rate * num_clusters)))

    # degenerate: <2 supported classes → untrustworthy local agreement
    if num_supported < 2:
        return -1e6 - float(2 - num_supported)

    # --- neighbors
    idx = _knn_indices(feats_np, k=k)

    # --- hard agreement + chance correction
    nn_labels = yhat[idx]                           # (N, k)
    SA = float((nn_labels == yhat[:, None]).mean())
    mean_p = probs_np.mean(axis=0)                  # predicted marginal
    P_match = float((mean_p ** 2).sum())            # chance match under marginal
    SA_adj = (SA - P_match) / max(eps, 1.0 - P_match)
    # SA_adj = SA

    # --- effective-class penalty (weight fixed to 1.0)
    effective_classes = 1.0 / max(eps, P_match)     # Hill number (q=2)
    target_for_penalty = max(2.0, float(required))
    shortfall = max(0.0, target_for_penalty - effective_classes) / target_for_penalty
    penalty_eff = shortfall  # weight = 1.0

    # --- coverage penalty
    coverage_shortfall = max(0.0, float(required - num_supported)) / float(required)
    penalty_cov = cov_penalty_weight * coverage_shortfall


    print(f'SA={SA}, SA_adj={SA_adj}, penalty_eff={-penalty_eff}, penalty_cov={-penalty_cov}')
    return SA_adj - (penalty_eff + penalty_cov)

# -----------------------------
# Alpha selection with early stop
# -----------------------------

def select_alpha_by_knn_self_agreement(
    model: nn.Module,
    state0: dict,
    taskvector,   # must have .apply_to(model, scaling_coef=alpha, strict=False)
    unlabeled_loader: DataLoader,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    *,
    num_clusters: int,
    alphas=(0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0),
    k: int = 10,
    bs_logits: int = 4096,
    device: str = "cuda",
    coverage_rate: float = 1.0,
    cov_penalty_weight: float = 1.0,
    hard_min_per_class: int | None = None,
    random_state: int = 0,
    decrease_patience: int = 10,
) -> float:
    """
    Select alpha that maximizes the adjusted, coverage-aware kNN self-agreement.

    Early stop if the score strictly decreases for `decrease_patience` consecutive
    steps (i.e., a window of length decrease_patience+1 is strictly descending).
    """
    best_alpha = None
    best_score = -float("inf")

    # rolling window of recent scores; length = patience + 1
    window = deque(maxlen=decrease_patience + 1)

    for a in alphas:
        # Reset -> apply correction
        model.load_state_dict(state0, strict=False)
        taskvector.apply_to(model, scaling_coef=a, strict=False)

        # Features & logits on unlabeled proxy
        feats, _ = extract_features(
            feature_extractor=feature_extractor,
            dataloader=unlabeled_loader,
            normalize=True,
            device=torch.device(device),
        )
        logits = logits_from_features(
            classifier=classifier, feats=feats, bs=bs_logits, device=device
        )
        probs = F.softmax(logits, dim=-1)

        # Adjusted score with coverage constraints
        score = knn_self_agreement_diversity(
            feats=feats,
            probs=probs,
            k=k,
            num_clusters=num_clusters,
            coverage_rate=coverage_rate,
            cov_penalty_weight=cov_penalty_weight,
            hard_min_per_class=hard_min_per_class,
            random_state=random_state,
        )
        
        print(f'Alpha={a}, Aggregate Score={score}')
        # track best
        if score > best_score:
            best_score = score
            best_alpha = float(a)

        # update window and check early-stop
        window.append(score)
        if len(window) == window.maxlen:
            # strictly decreasing if all diffs < 0
            if np.all(np.diff(np.array(window)) < 0):
                break

    return best_alpha
