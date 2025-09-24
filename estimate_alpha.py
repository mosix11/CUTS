import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        
        # hard coding for DinoV3 (kept as-is)
        if not isinstance(z, torch.Tensor):
            z = getattr(z, "pooler_output", None)
            if z is None:
                z = z.last_hidden_state[:, 0, :]
        
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


def knn_self_agreement(feats: torch.Tensor, probs: torch.Tensor, k: int = 10):
    from sklearn.neighbors import NearestNeighbors
    yhat = probs.argmax(dim=1).numpy()
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(feats)), algorithm='auto').fit(feats)
    _, idx = nbrs.kneighbors(feats, return_distance=True)
    idx = idx[:, 1:]  # drop self
    nn_labels = yhat[idx]
    return float((nn_labels == yhat[:, None]).mean())


def select_alpha_by_knn_self_agreement(
    model: nn.Module,
    state0: dict,
    taskvector,  # must have .apply_to(model, scaling_coef=alpha, strict=False)
    unlabeled_loader: DataLoader,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    alphas=(0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0),
    bs_logits: int = 4096,
    device: str = "cuda",
) -> float:
    """
    Select the correction coefficient alpha that maximizes kNN self-agreement
    on unlabeled data. For each alpha: reset model -> apply taskvector -> extract
    features -> compute logits -> compute kNN self-agreement. Returns the alpha
    with the highest kNN agreement (float).
    """
    best_alpha = None
    best_knn = -float("inf")

    for a in alphas:
        # Reset -> apply correction
        model.load_state_dict(state0, strict=False)
        taskvector.apply_to(model, scaling_coef=a, strict=False)

        # Features & logits
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

        # kNN self-agreement
        knn_val = knn_self_agreement(feats, probs)

        if knn_val > best_knn:
            best_knn = knn_val
            best_alpha = float(a)

    return best_alpha
