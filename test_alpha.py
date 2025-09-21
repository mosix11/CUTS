import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
import matplotlib.pyplot as plt
from transformers import AutoModel

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
        feats: (N, D) float32 tensor on cpu
        labels: (N,) long tensor on cpu
    """
    feature_extractor.to(device).eval()
    feats, labels = [], []
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        x, y = prepare_batch(batch, device)[:2]
        z = feature_extractor(x)  # (B, D)
        if normalize:
            z = F.normalize(z, dim=1)
        feats.append(z.detach().cpu())
        labels.append(y.detach().cpu().long())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

# -----------------------------
# Unlabeled metrics utilities
# -----------------------------
@torch.no_grad()
def softmax_probs(logits, temp=1.0):
    return F.softmax(logits / temp, dim=-1)

def entropy_t(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0).cpu()
    return -(p * p.log()).sum(dim=-1)

def zscore(arr):
    arr = np.asarray(arr, float)
    return (arr - arr.mean()) / (arr.std() + 1e-12)

# -----------------------------
# v2 transforms for already-normalized tensors
# -----------------------------
class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std  = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x):
        return x * self.std + self.mean

class Renormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std  = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x):
        return (x - self.mean) / self.std

def build_weak_transform_v2(
    image_size: int,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    allow_color: bool = False,
):
    geom = [
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=7,
            translate=(0.02, 0.02),
            scale=(0.97, 1.03),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ]
    if not allow_color:
        return transforms.Compose(geom)
    color = [
        Denormalize(mean, std),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        Renormalize(mean, std),
    ]
    return transforms.Compose(geom + color)

def make_two_pipelines(image_size, mean, std, allow_color=False):
    t1 = build_weak_transform_v2(image_size, mean, std, allow_color=allow_color)
    t2 = build_weak_transform_v2(image_size, mean, std, allow_color=allow_color)
    return t1, t2

def two_augs(
    x: torch.Tensor,
    *,
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    allow_color: bool = False,
    _pipelines_cache: dict = {},
):
    """
    x: (B, C, H, W) float32 tensor, ALREADY normalized with given mean/std.
    Returns (x1, x2) with the SAME normalization and shape.
    """
    key = (image_size, tuple(mean), tuple(std), allow_color)
    if key not in _pipelines_cache:
        _pipelines_cache[key] = make_two_pipelines(image_size, mean, std, allow_color)
    t1, t2 = _pipelines_cache[key]
    x1 = t1(x)  # v2 supports batched tensors
    x2 = t2(x)
    return x1.to(x.device, non_blocking=True), x2.to(x.device, non_blocking=True)

# -----------------------------
# Classifier logits from features
# -----------------------------
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
# Remaining scores: kNN self-agreement
# -----------------------------
def knn_self_agreement(feats: torch.Tensor, probs: torch.Tensor, k: int = 10):
    from sklearn.neighbors import NearestNeighbors
    yhat = probs.argmax(dim=1).numpy()
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(feats)), algorithm='auto').fit(feats)
    _, idx = nbrs.kneighbors(feats, return_distance=True)
    idx = idx[:, 1:]  # drop self
    nn_labels = yhat[idx]
    return float((nn_labels == yhat[:, None]).mean())

# -----------------------------
# Aug-consistency (S1/S4) using dataloader
# -----------------------------
@torch.no_grad()
def collect_aug_consistency_with_loader(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
):
    """
    Computes on unlabeled data:
      - L_cons: average KL between p(y|t1(x)) and p(y|t2(x))
      - H_marg: entropy of dataset-level marginal p(y)
      - H_cond: mean per-sample conditional entropy
    """
    feature_extractor = feature_extractor.to(device).eval()
    classifier = classifier.to(device).eval()

    kl_sum, N = 0.0, 0
    marg_sum = None
    cond_H_sum = 0.0

    for batch in tqdm(dataloader, desc="Aug-consistency", leave=False):
        x, _ = prepare_batch(batch, device)[:2]
        # Use current spatial size
        x1, x2 = two_augs(x, image_size=x.shape[-1])

        f1 = feature_extractor(x1)
        # hard coding for DinoV3
        if not isinstance(feature_extractor, torch.Tensor):
            f1 = getattr(f1, "pooler_output", None)
            if f1 is None:
                f1 = f1.last_hidden_state[:, 0, :]
        
        f2 = feature_extractor(x2)
        # hard coding for DinoV3
        if not isinstance(feature_extractor, torch.Tensor):
            f2 = getattr(f2, "pooler_output", None)
            if f2 is None:
                f2 = f2.last_hidden_state[:, 0, :]
        z1 = classifier(f1)
        z2 = classifier(f2)
        p1, p2 = softmax_probs(z1), softmax_probs(z2)
        kl = (p1 * (p1.clamp(1e-12, 1).log() - p2.clamp(1e-12, 1).log())).sum(dim=-1)
        kl_sum += kl.sum().item()
        cond_H_sum += (entropy_t(p1) + entropy_t(p2)).sum().item() / 2.0

        if marg_sum is None:
            marg_sum = p1.sum(dim=0) + p2.sum(dim=0)
        else:
            marg_sum += p1.sum(dim=0) + p2.sum(dim=0)

        N += x.size(0)

    L_cons = kl_sum / N
    p_marg = (marg_sum / (2 * N)).cpu().numpy()
    H_marg = float(-(p_marg * np.log(np.clip(p_marg, 1e-12, 1))).sum())
    H_cond = float(cond_H_sum / N)
    return L_cons, H_marg, H_cond

# -----------------------------
# Main alpha selection (S1, S3, S4 only)
# -----------------------------
def select_alpha_star(
    model: nn.Module,
    state0: dict,
    taskvector,  # must have .apply_to(model, scaling_coef=alpha, strict=False)
    unlabeled_loader: DataLoader,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    alphas=(0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0),
    # weights now only (w1, w3, w4): S1, kNN, S4
    w=(2.0, 1.5, 1.0),
    lambda_cons=1.0,
    beta_mi=0.3,
    bs_logits=4096,
    device="cuda",
):
    """
    J(alpha) = w1 * z(S1) + w3 * z(kNN) + w4 * z(S4)
    Returns:
      best_record (dict with max J),
      records (list of dicts over alphas),
      alpha_best (dict with best alpha per metric + by J)
    """
    records = []

    for a in alphas:
        # Reset -> apply correction
        model.load_state_dict(state0, strict=False)
        taskvector.apply_to(model, scaling_coef=a, strict=False)

        # S1 & S4 (consistency/diversity, entropy gap)
        L_cons, H_marg, H_cond = collect_aug_consistency_with_loader(
            feature_extractor, classifier, unlabeled_loader, device
        )
        S1 = H_marg - lambda_cons * L_cons
        S4 = H_marg - beta_mi * H_cond

        # Features & logits -> kNN self-agreement (S3)
        feats, _ = extract_features(
            feature_extractor=feature_extractor,
            dataloader=unlabeled_loader,
            normalize=True,
            device=torch.device(device),
        )
        logits = logits_from_features(
            classifier=classifier, feats=feats, bs=bs_logits, device=device
        )
        kNN = knn_self_agreement(feats, F.softmax(logits, dim=-1))

        records.append(dict(alpha=a, S1=S1, kNN=kNN, S4=S4))

    # Arrays
    alphas_arr = np.array([r['alpha'] for r in records], dtype=float)
    S1_arr     = np.array([r['S1']   for r in records], dtype=float)
    kNN_arr    = np.array([r['kNN']  for r in records], dtype=float)
    S4_arr     = np.array([r['S4']   for r in records], dtype=float)

    # z-scores and J
    S1z = zscore(S1_arr)
    S3z = zscore(kNN_arr)
    S4z = zscore(S4_arr)
    w1, w3, w4 = w
    J = w1*S1z + w3*S3z + w4*S4z

    # attach z-scores and J into records
    for r, s1, s3, s4, j in zip(records, S1z, S3z, S4z, J):
        r.update(S1z=s1, S3z=s3, S4z=s4, J=j)

    # Best by J and per metric (raw)
    idx_J   = int(np.argmax(J))
    idx_S1  = int(np.argmax(S1_arr))
    idx_kNN = int(np.argmax(kNN_arr))
    idx_S4  = int(np.argmax(S4_arr))

    alpha_best = {
        "alpha_J":   float(alphas_arr[idx_J]),
        "J_value":   float(J[idx_J]),
        "alpha_S1":  float(alphas_arr[idx_S1]),
        "S1_value":  float(S1_arr[idx_S1]),
        "alpha_kNN": float(alphas_arr[idx_kNN]),
        "kNN_value": float(kNN_arr[idx_kNN]),
        "alpha_S4":  float(alphas_arr[idx_S4]),
        "S4_value":  float(S4_arr[idx_S4]),
    }

    best = records[idx_J]
    return best, records, alpha_best

# -----------------------------
# Plotting (only S1, S3, S4)
# -----------------------------
def plot_alpha_metrics(
    records,
    weights=(2.0, 1.5, 1.0),      # (w1, w3, w4)
    title_prefix="Alpha Search",
    save_prefix=None,
    show=True,
    alpha_best: dict = None,      # optional: pass dict returned by select_alpha_star
):
    """
    Plots:
      Fig 1: Weighted z-scored components (w1*S1z, w3*S3z, w4*S4z) + J(alpha),
             with vertical lines at best α for J, S1, kNN, S4.
      Fig 2: Raw metrics curves (S1, kNN, S4) with the same vertical lines.
    """
    # order by alpha for plotting
    alphas = np.array([r["alpha"] for r in records], dtype=float)
    order = np.argsort(alphas)
    alphas = alphas[order]

    def arr(key):
        return np.array([records[i][key] for i in order], dtype=float)

    # raw
    S1_raw = arr("S1")
    kNN_raw = arr("kNN")
    S4_raw = arr("S4")

    # z-scores: use precomputed if present, else compute
    if "S1z" in records[0]:
        S1z = arr("S1z"); S3z = arr("S3z"); S4z = arr("S4z")
        J = arr("J") if "J" in records[0] else None
    else:
        S1z = zscore(S1_raw)
        S3z = zscore(kNN_raw)
        S4z = zscore(S4_raw)
        w1, w3, w4 = weights
        J = w1*S1z + w3*S3z + w4*S4z

    w1, w3, w4 = weights
    C1 = w1 * S1z
    C3 = w3 * S3z
    C4 = w4 * S4z

    # Compute best alphas if not provided
    if alpha_best is None:
        idx_J   = int(np.argmax(J))
        idx_S1  = int(np.argmax(S1_raw))
        idx_kNN = int(np.argmax(kNN_raw))
        idx_S4  = int(np.argmax(S4_raw))
        alpha_best = {
            "alpha_J":   float(alphas[idx_J]),
            "alpha_S1":  float(alphas[idx_S1]),
            "alpha_kNN": float(alphas[idx_kNN]),
            "alpha_S4":  float(alphas[idx_S4]),
        }
    alpha_J   = alpha_best.get("alpha_J")
    alpha_S1  = alpha_best.get("alpha_S1")
    alpha_kNN = alpha_best.get("alpha_kNN")
    alpha_S4  = alpha_best.get("alpha_S4")

    # best J value for title
    j_best_idx = int(np.argmax(J))
    alpha_star = alphas[j_best_idx]
    j_best = J[j_best_idx]

    # ---- Figure 1: weighted z-components + J ----
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, C1, marker="o", label=f"w1*S1z (consistency-diversity) [w1={w1}]")
    plt.plot(alphas, C3, marker="o", label=f"w3*S3z (kNN agreement) [w3={w3}]")
    plt.plot(alphas, C4, marker="o", label=f"w4*S4z (entropy gap) [w4={w4}]")
    plt.plot(alphas, J,  marker="o", linestyle="--", label="J(α) total")

    # vertical markers
    plt.axvline(alpha_J,   linestyle=":",  linewidth=1.2, label=f"best α by J = {alpha_J:.4g}")
    plt.axvline(alpha_S1,  linestyle="--", linewidth=1.0, label=f"best α by S1 = {alpha_S1:.4g}")
    plt.axvline(alpha_kNN, linestyle="-.", linewidth=1.0, label=f"best α by kNN = {alpha_kNN:.4g}")
    plt.axvline(alpha_S4,  linestyle=(0,(3,1)), linewidth=1.0, label=f"best α by S4 = {alpha_S4:.4g}")

    plt.title(f"{title_prefix}: Weighted z-scored components & J\nbest α(J) = {alpha_star:.4g}, J = {j_best:.3f}")
    plt.xlabel("alpha (scaling coef)")
    plt.ylabel("score (arb. units)")
    plt.grid(True, alpha=0.3)
    # dedupe legend entries if lines share same α
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), loc="best")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_weighted.png", dpi=150)
    if show: plt.show()
    else: plt.close()

    # ---- Figure 2: raw metrics ----
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, S1_raw, marker="o", label="S1 = H_marg - λ*KL(t1||t2) (↑)")
    plt.plot(alphas, kNN_raw, marker="o", label="kNN self-agreement (↑)")
    plt.plot(alphas, S4_raw, marker="o", label="S4 = H_marg - β*H_cond (↑)")

    plt.axvline(alpha_J,   linestyle=":",  linewidth=1.2, label=f"best α by J = {alpha_J:.4g}")
    plt.axvline(alpha_S1,  linestyle="--", linewidth=1.0, label=f"best α by S1 = {alpha_S1:.4g}")
    plt.axvline(alpha_kNN, linestyle="-.", linewidth=1.0, label=f"best α by kNN = {alpha_kNN:.4g}")
    plt.axvline(alpha_S4,  linestyle=(0,(3,1)), linewidth=1.0, label=f"best α by S4 = {alpha_S4:.4g}")

    plt.title(f"{title_prefix}: Raw metrics (higher is better)")
    plt.xlabel("alpha (scaling coef)")
    plt.ylabel("raw score")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), loc="best")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_raw.png", dpi=150)
    if show: plt.show()
    else: plt.close()

    # Compact table
    header = ("alpha", "J", "w1*S1z", "w3*S3z", "w4*S4z", "S1_raw", "kNN", "S4_raw")
    rows = np.column_stack([alphas, J, w1*S1z, w3*S3z, w4*S4z, S1_raw, kNN_raw, S4_raw])
    with np.printoptions(precision=4, suppress=True):
        print("\n[Alpha metrics summary]")
        print(" | ".join(f"{h:>8s}" for h in header))
        for r in rows:
            print(" | ".join(f"{v:8.4f}" for v in r))

    return dict(
        alpha_J=float(alpha_J),
        alpha_S1=float(alpha_S1),
        alpha_kNN=float(alpha_kNN),
        alpha_S4=float(alpha_S4),
        J_best=float(j_best),
    )
