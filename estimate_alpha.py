# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm


# def prepare_batch(batch, device):
#     return [t.to(device, non_blocking=True) for t in batch]

# @torch.no_grad()
# def extract_features(
#     feature_extractor: nn.Module,
#     dataloader: DataLoader,
#     normalize: bool = True,
#     device: torch.device = torch.device('cpu'),
# ):
#     """
#     Returns:
#         feats: (N, D) float32 tensor on cpu
#         labels: (N,) long tensor on cpu
#     """
#     feature_extractor.to(device).eval()
#     feats, labels = [], []
#     for batch in tqdm(dataloader, desc="Extracting features", leave=False):
#         x, y = prepare_batch(batch, device)[:2]
#         z = feature_extractor(x)  # (B, D)
        
#         # hard coding for DinoV3 (kept as-is)
#         if not isinstance(z, torch.Tensor):
#             z = getattr(z, "pooler_output", None)
#             if z is None:
#                 z = z.last_hidden_state[:, 0, :]
        
#         if normalize:
#             z = F.normalize(z, dim=1)
#         feats.append(z.detach().cpu())
#         labels.append(y.detach().cpu().long())
#     feats = torch.cat(feats, dim=0)
#     labels = torch.cat(labels, dim=0)
#     return feats, labels


# @torch.no_grad()
# def logits_from_features(classifier: nn.Module, feats: torch.Tensor, bs: int = 4096, device: str = "cuda"):
#     """
#     Args:
#         feats: (N, D) on CPU
#     Returns:
#         logits: (N, K) on CPU (float32)
#     """
#     classifier = classifier.to(device).eval()
#     outs = []
#     N = feats.size(0)
#     for i in range(0, N, bs):
#         fb = feats[i:i+bs].to(device, non_blocking=True)
#         outs.append(classifier(fb).detach().float().cpu())
#     return torch.cat(outs, dim=0)


# def knn_self_agreement(feats: torch.Tensor, probs: torch.Tensor, k: int = 10):
#     from sklearn.neighbors import NearestNeighbors
#     yhat = probs.argmax(dim=1).numpy()
#     nbrs = NearestNeighbors(n_neighbors=min(k+1, len(feats)), algorithm='auto').fit(feats)
#     _, idx = nbrs.kneighbors(feats, return_distance=True)
#     idx = idx[:, 1:]  # drop self
#     nn_labels = yhat[idx]
#     return float((nn_labels == yhat[:, None]).mean())


# def select_alpha_by_knn_self_agreement(
#     model: nn.Module,
#     state0: dict,
#     taskvector,  # must have .apply_to(model, scaling_coef=alpha, strict=False)
#     unlabeled_loader: DataLoader,
#     feature_extractor: nn.Module,
#     classifier: nn.Module,
#     alphas=(0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0),
#     bs_logits: int = 4096,
#     device: str = "cuda",
# ) -> float:
#     """
#     Select the correction coefficient alpha that maximizes kNN self-agreement
#     on unlabeled data. For each alpha: reset model -> apply taskvector -> extract
#     features -> compute logits -> compute kNN self-agreement. Returns the alpha
#     with the highest kNN agreement (float).
#     """
#     best_alpha = None
#     best_knn = -float("inf")

#     for a in alphas:
#         # Reset -> apply correction
#         model.load_state_dict(state0, strict=False)
#         taskvector.apply_to(model, scaling_coef=a, strict=False)

#         # Features & logits
#         feats, _ = extract_features(
#             feature_extractor=feature_extractor,
#             dataloader=unlabeled_loader,
#             normalize=True,
#             device=torch.device(device),
#         )
#         logits = logits_from_features(
#             classifier=classifier, feats=feats, bs=bs_logits, device=device
#         )
#         probs = F.softmax(logits, dim=-1)

#         # kNN self-agreement
#         knn_val = knn_self_agreement(feats, probs)

#         if knn_val > best_knn:
#             best_knn = knn_val
#             best_alpha = float(a)

#     return best_alpha


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def prepare_batch(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]

@torch.no_grad()
def extract_features(
    feature_extractor: torch.nn.Module,
    dataloader: DataLoader,
    normalize: bool = True,
    device: torch.device = torch.device('cpu'),
):
    feature_extractor.to(device).eval()
    feats, labels = [], []
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        x, y = prepare_batch(batch, device)[:2]
        z = feature_extractor(x)
        # specific handling for models that return dict/transformers outputs
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
def logits_from_features(classifier: torch.nn.Module, feats: torch.Tensor, bs: int = 4096, device: str = "cuda"):
    classifier = classifier.to(device).eval()
    outs = []
    N = feats.size(0)
    for i in range(0, N, bs):
        fb = feats[i:i+bs].to(device, non_blocking=True)
        outs.append(classifier(fb).detach().float().cpu())
    return torch.cat(outs, dim=0)

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
    target_classes: int,
    penalty_weight: float = 1.0,
    gamma_nmi: float = 0.2,
    hard_min_per_class: int | None = None,
    use_soft_agreement: bool = False,
    kmeans_sample: int | None = None,
    random_state: int = 0,
) -> float:
    """
    Unsupervised score that (i) uses chance-corrected kNN self-agreement,
    (ii) strongly penalizes if the predicted class support is < target_classes,
    and (iii) optionally rewards alignment with k-means(C=target_classes) clusters.

    Returns a scalar to maximize.
    """
    eps = 1e-8
    N, K = probs.shape
    feats_np = feats.numpy()
    probs_np = probs.numpy()
    yhat = probs_np.argmax(axis=1)

    # --- diversity checks
    counts = np.bincount(yhat, minlength=K)
    if hard_min_per_class is None:
        hard_min_per_class = min(k + 1, max(3, k))  # enough to form local neighborhoods
    num_classes_with_support = int((counts >= hard_min_per_class).sum())

    # Hard guard: if we can't even see target_classes with sufficient support, nuke the score
    if num_classes_with_support < min(target_classes, K):
        return -1e9 - float(target_classes - num_classes_with_support)

    # --- kNN neighbors
    idx = _knn_indices(feats_np, k=k)

    # --- agreement term (hard or soft), then chance correction
    if use_soft_agreement:
        # soft: average prob-vector similarity with neighbors
        # expected chance similarity = sum_c (mean_p[c])^2
        neigh_probs = probs_np[idx]                              # (N, k, K)
        dot_soft = (neigh_probs * probs_np[:, None, :]).sum(-1)  # (N, k)
        SA = float(dot_soft.mean())
        mean_p = probs_np.mean(axis=0)
        P_match = float((mean_p ** 2).sum())
    else:
        # hard: equality of argmax labels
        nn_labels = yhat[idx]                                    # (N, k)
        SA = float((nn_labels == yhat[:, None]).mean())
        # chance that two random samples share label under predicted marginal
        mean_p = probs_np.mean(axis=0)
        P_match = float((mean_p ** 2).sum())

    SA_adj = (SA - P_match) / max(eps, 1.0 - P_match)

    # --- soft penalty via "effective number of classes"
    # E = 1 / sum p_c^2; smaller than target indicates collapse/imbalance
    effective_classes = 1.0 / max(eps, P_match)
    shortfall = max(0.0, target_classes - effective_classes) / max(1.0, target_classes)
    penalty = penalty_weight * shortfall

    # --- optional structure bonus: NMI(yhat, kmeans_C)
    nmi_bonus = 0.0
    if gamma_nmi > 0.0 and target_classes >= 2 and N >= target_classes:
        from sklearn.cluster import KMeans
        from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

        if kmeans_sample is not None and N > kmeans_sample:
            rng = np.random.default_rng(random_state)
            sel = rng.choice(N, size=kmeans_sample, replace=False)
            Xk = feats_np[sel]
            yk = yhat[sel]
        else:
            Xk = feats_np
            yk = yhat

        nclust = min(target_classes, len(Xk))
        if nclust >= 2:
            km = KMeans(n_clusters=nclust, n_init=10, random_state=random_state)
            km_labels = km.fit_predict(Xk)
            nmi_bonus = float(NMI(yk, km_labels)) * gamma_nmi

    return SA_adj - penalty + nmi_bonus

def select_alpha_by_knn_self_agreement(
    model: torch.nn.Module,
    state0: dict,
    taskvector,  # must have .apply_to(model, scaling_coef=alpha, strict=False)
    unlabeled_loader: DataLoader,
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    *,
    target_classes: int,
    alphas=(0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0),
    k: int = 10,
    bs_logits: int = 4096,
    device: str = "cuda",
    penalty_weight: float = 1.0,
    gamma_nmi: float = 0.2,
    hard_min_per_class: int | None = None,
    use_soft_agreement: bool = False,
    kmeans_sample: int | None = None,
    random_state: int = 0,
) -> float:
    """
    Select alpha that maximizes the adjusted, diversity-aware kNN self-agreement.
    """
    best_alpha = None
    best_score = -float("inf")

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

        # Adjusted score with diversity constraints
        score = knn_self_agreement_diversity(
            feats=feats,
            probs=probs,
            k=k,
            target_classes=target_classes,
            penalty_weight=penalty_weight,
            gamma_nmi=gamma_nmi,
            hard_min_per_class=hard_min_per_class,
            use_soft_agreement=use_soft_agreement,
            kmeans_sample=kmeans_sample,
            random_state=random_state,
        )

        if score > best_score:
            best_score = score
            best_alpha = float(a)

    return best_alpha
