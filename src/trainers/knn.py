import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm


def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch

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
            z = F.normalize(z, dim=1)    # cosine-friendly
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
