import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm


def prepare_batch(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]

@torch.no_grad()
def extract_features(
    feature_extractor:nn.Module,
    dataloader:DataLoader,
    device:torch.device = torch.device('cpu'),
    normalize:bool = True
):
    """
    Returns:
        feats: (N, D) float32 tensor on device='cpu'
        labels: (N,) long tensor on cpu
    """
    feature_extractor.to(device)
    feature_extractor.eval()
    feats, labels = [], []
    
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        
        batch = prepare_batch(batch, device)
        x, y = batch[:2]
        
        z = feature_extractor(x)      # shape (B, D)
        if normalize:
            z = F.normalize(z, dim=1)    
        feats.append(z.detach().cpu())
        labels.append(y.detach().cpu().long())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


@torch.no_grad()
def knn_predict(
    query_feats:torch.Tensor,
    ref_feats:torch.Tensor,
    ref_labels:torch.Tensor,
    k:int = 20,
    T:int = 0.07,
    weighted:bool = True,
    batch_size:int = 2048,
    device:torch.device = torch.device('cpu')
):
    """
    k-NN prediction with cosine similarity.
    - If weighted=True: soft voting with exp(sim/T) (kNN monitor style).
    - If weighted=False: majority vote.
    Args:
        query_feats: (Nq, D) cpu tensor (L2-normalized if cosine)
        ref_feats:   (Nr, D) cpu tensor (L2-normalized if cosine)
        ref_labels:  (Nr,) cpu long tensor
    Returns:
        preds: (Nq,) cpu long tensor
    """
    # Move reference feats to device once (if it fits). Keep labels on cpu.
    ref_feats_dev = ref_feats.to(device, non_blocking=True)
    query_preds = []

    # Precompute one-hot for fast weighted voting
    num_classes = int(ref_labels.max().item()) + 1
    # (Nr, C) one-hot on cpu; we’ll index rows and move the slice per batch
    ref_onehot = torch.zeros(ref_feats.size(0), num_classes, dtype=torch.float32)
    ref_onehot[torch.arange(ref_feats.size(0)), ref_labels] = 1.0

    for start in tqdm(range(0, query_feats.size(0), batch_size),
                      desc="kNN predict", leave=False):
        end = min(start + batch_size, query_feats.size(0))
        q = query_feats[start:end].to(device, non_blocking=True)  # (B, D)

        # cosine similarity because features are normalized
        # sim: (B, Nr)
        sim = q @ ref_feats_dev.T

        # Top-k neighbors (values and indices)
        vals, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)  # (B, k)

        if weighted:
            weights = torch.exp(vals / T)  # (B, k)
        else:
            weights = torch.ones_like(vals)

        # gather labels as one-hot for the k neighbors, then weighted sum
        # neighbor_onehot: (B, k, C) on cpu; bring to device minimally
        neigh_oh = ref_onehot.index_select(0, idx.flatten().cpu()).view(idx.size(0), k, num_classes)
        neigh_oh = neigh_oh.to(device, non_blocking=True)

        # class_scores: (B, C)
        class_scores = (neigh_oh * weights.unsqueeze(-1)).sum(dim=1)

        pred = class_scores.argmax(dim=1).detach().cpu()
        query_preds.append(pred)

    return torch.cat(query_preds, dim=0)


@torch.no_grad()
def knn_eval(feature_extractor:nn.Module,
             train_dl:DataLoader,
             test_dl:DataLoader,
             k:int = 20,
             T:float = 0.07,
             weighted:bool = True,
             device:torch.device = torch.device('cpu'),
             normalize:bool = True,
             batch_size_predict:int = 2048
    ) -> dict:
    """
    Full pipeline:
      1) extract train features+labels (reference set)
      2) extract test features+labels (queries)
      3) k-NN prediction
      4) report top-1 accuracy
    """
    # 1) reference set (train)
    train_feats, train_labels = extract_features(feature_extractor, train_dl, device=device, normalize=normalize)

    # 2) query set (test)
    test_feats, test_labels = extract_features(feature_extractor, test_dl, device=device, normalize=normalize)

    # 3) predict via k-NN
    preds = knn_predict(
        query_feats=test_feats,
        ref_feats=train_feats,
        ref_labels=train_labels,
        k=k,
        T=T,
        weighted=weighted,
        batch_size=batch_size_predict,
        device=device,
    )

    # 4) accuracy
    acc = (preds == test_labels).float().mean().item() * 100.0
    return {"top1_acc": acc, "k": k, "T": T, "weighted": weighted}



@torch.no_grad()
def compute_class_prototypes(
    feats: torch.Tensor,           # (N, D) on CPU
    labels: torch.Tensor,          # (N,)   on CPU
    normalize: bool = True,
):
    """
    Compute per-class centroids (prototypes).
    Handles non-contiguous labels by returning the 'classes' tensor that maps row->label_id.

    Returns:
        prototypes: (C, D) float32 on CPU
        classes:    (C,)   long   label IDs corresponding to rows of 'prototypes'
        counts:     (C,)   long   number of samples per class
    """
    classes, inv = labels.unique(sorted=True, return_inverse=True)  # inv maps each sample -> row in [0..C-1]
    C = classes.numel()
    D = feats.size(1)

    # Sum features per class via scatter_add
    sums = torch.zeros(C, D, dtype=feats.dtype, device=feats.device)
    sums.index_add_(0, inv, feats)  # accumulate per class

    counts = torch.bincount(inv, minlength=C).to(feats.dtype).unsqueeze(1)  # (C,1)
    prototypes = sums / counts.clamp_min(1.0)  # safe divide

    if normalize:
        prototypes = F.normalize(prototypes, dim=1)

    return prototypes.cpu(), classes.cpu(), counts.squeeze(1).cpu().long()


@torch.no_grad()
def ncm_predict(
    query_feats: torch.Tensor,     # (Nq, D) CPU (normalized if metric='cosine')
    prototypes: torch.Tensor,      # (C,  D) CPU (normalized if metric='cosine')
    classes: torch.Tensor,         # (C,)   CPU (actual label IDs for rows in 'prototypes')
    metric: str = "cosine",        # 'cosine' or 'euclidean'
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
):
    """
    Predict by nearest class prototype.
    Returns:
        preds: (Nq,) CPU long tensor of *original* label IDs.
    """
    # Move prototypes once
    P = prototypes.to(device, non_blocking=True)  # (C, D)
    preds = []

    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    # Precompute norms if Euclidean
    if metric == "euclidean":
        P_sq = (P * P).sum(dim=1, keepdim=True)  # (C,1)

    for start in tqdm(range(0, query_feats.size(0), batch_size),
                      desc="NCM predict", leave=False):
        end = min(start + batch_size, query_feats.size(0))
        Q = query_feats[start:end].to(device, non_blocking=True)  # (B, D)

        if metric == "cosine":
            # Assumes (optionally) L2-normalized feats & prototypes: higher is better
            sims = Q @ P.T  # (B, C)
            idx = sims.argmax(dim=1)  # nearest = highest cosine similarity
        else:
            # d^2(q,p) = ||q||^2 + ||p||^2 - 2 q·p ; choose argmin
            Q_sq = (Q * Q).sum(dim=1, keepdim=True)  # (B,1)
            cross = Q @ P.T                           # (B,C)
            d2 = Q_sq + P_sq.T - 2.0 * cross         # (B,C)
            idx = d2.argmin(dim=1)

        preds.append(classes[idx.detach().cpu()])

    return torch.cat(preds, dim=0)


@torch.no_grad()
def ncm_eval(
    feature_extractor: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    metric: str = "cosine",        # 'cosine' or 'euclidean'
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
    batch_size_predict: int = 4096,
):
    """
    Full pipeline:
      1) extract train features+labels (reference set)
      2) compute class prototypes
      3) extract test features+labels (queries)
      4) nearest-prototype prediction
      5) top-1 accuracy
    """
    # 1) reference set
    train_feats, train_labels = extract_features(
        feature_extractor, train_dl, device=device, normalize=normalize
    )

    # 2) prototypes
    prototypes, classes, counts = compute_class_prototypes(
        train_feats, train_labels, normalize=(normalize and metric == "cosine")
    )

    # 3) queries
    test_feats, test_labels = extract_features(
        feature_extractor, test_dl, device=device, normalize=normalize
    )

    # 4) predict
    preds = ncm_predict(
        query_feats=test_feats,
        prototypes=prototypes,
        classes=classes,
        metric=metric,
        batch_size=batch_size_predict,
        device=device,
    )

    # 5) accuracy
    top1 = (preds == test_labels).float().mean().item() * 100.0

    return {
        "top1_acc": top1,
        "metric": metric,
        "normalize": normalize,
        "num_classes_in_train": int(classes.numel()),
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
    }
    
    
@torch.no_grad()
def knn_ncm_eval(
    feature_extractor: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    *,
    # shared
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
    # k-NN params
    k: int = 20,
    T: float = 0.07,
    weighted: bool = True,
    knn_batch_size: int = 2048,
    # NCM params
    ncm_metric: str = "cosine",      # 'cosine' or 'euclidean'
    ncm_batch_size: int = 4096,
):
    """
    Extract features once, then evaluate both k-NN and NCM.

    Returns:
        {
          'knn': {'top1_acc': ..., 'k': ..., 'T': ..., 'weighted': ...},
          'ncm': {
              'top1_acc': ..., 'metric': ..., 'normalize': ...,
              'num_classes_in_train': C, 'class_counts': {label: count, ...}
          },
          'shared': {'normalize': normalize}
        }
    """
    # 1) Extract features ONCE
    train_feats, train_labels = extract_features(
        feature_extractor, train_dl, device=device, normalize=normalize
    )
    test_feats, test_labels = extract_features(
        feature_extractor, test_dl, device=device, normalize=normalize
    )

    # 2) k-NN on the extracted features
    knn_preds = knn_predict(
        query_feats=test_feats,
        ref_feats=train_feats,
        ref_labels=train_labels,
        k=k, T=T, weighted=weighted,
        batch_size=knn_batch_size,
        device=device,
    )
    knn_acc = (knn_preds == test_labels).float().mean().item() * 100.0

    # 3) NCM on the same features
    #    If using cosine NCM, keep prototypes normalized; for euclidean, use raw means.
    proto_norm = (normalize and ncm_metric == "cosine")
    prototypes, classes, counts = compute_class_prototypes(
        train_feats, train_labels, normalize=proto_norm
    )
    ncm_preds = ncm_predict(
        query_feats=test_feats,
        prototypes=prototypes,
        classes=classes,
        metric=ncm_metric,
        batch_size=ncm_batch_size,
        device=device,
    )
    ncm_acc = (ncm_preds == test_labels).float().mean().item() * 100.0

    # 4) Package results
    return {
        "knn": {
            "top1_acc": knn_acc,
            "k": k,
            "T": T,
            "weighted": weighted,
        },
        "ncm": {
            "top1_acc": ncm_acc,
            "metric": ncm_metric,
            "normalize": proto_norm,
            "num_classes_in_train": int(classes.numel()),
            "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        },
        "shared": {"normalize": normalize},
    }