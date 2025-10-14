import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchdr
from typing import List
import datamapplot

from typing import Dict, Any, Optional, Tuple, List

from pathlib import Path
import imageio.v2 as imageio
import pickle
import gc

def _build_dataloader(dataset, batch_size):
    return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )


def prepare_batch(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]

@torch.no_grad()
def extract_features(
    feature_extractor:nn.Module,
    dataloader:DataLoader,
    normalize:bool = True,
    device:torch.device = torch.device('cpu'),
):
    """
    Returns:
        feats: (N, D) float32 tensor on device='cpu'
        labels: (N,) long tensor on cpu
    """
    feature_extractor.to(device)
    feature_extractor.eval()
    feats, labels, is_noisy = [], [], []
    
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        
        batch = prepare_batch(batch, device)
        if len(batch) == 4:
            x, y, ind, isn = batch
        else:
            x, y, ind = batch
            isn = torch.zeros_like(y)
        z = feature_extractor(x)      # shape (B, D)
        # hard coding for DinoV3 (kept as-is)
        if not isinstance(z, torch.Tensor):
            z = getattr(z, "pooler_output", None)
            if z is None:
                z = z.last_hidden_state[:, 0, :]
        if normalize:
            z = F.normalize(z, dim=1)    
        feats.append(z.detach().cpu())
        labels.append(y.detach().cpu().long())
        is_noisy.append(isn.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    is_noisy = torch.cat(is_noisy, dim=0)
    return feats, labels, is_noisy


# ------- helpers (rotation-only alignment) -------

def _labels_to_names(labels: torch.Tensor, class_names: List[str]) -> np.ndarray:
    idx2name = {int(i): n for i, n in enumerate(class_names)}
    return np.array([idx2name[int(i)] for i in labels.detach().cpu().numpy()])

def _pca_coords(features: torch.Tensor, device: torch.device) -> np.ndarray:
    z = torchdr.PCA(n_components=2, device=device).fit_transform(features)
    return z.detach().cpu().numpy()  # PCA scores, already centered (≈ zero mean)

def _orthogonal_matrix(Y: np.ndarray, X: np.ndarray, allow_reflection: bool = True) -> np.ndarray:
    """
    Compute the 2x2 orthogonal matrix R (rotation ± reflection) that best aligns Y to X.
    IMPORTANT: We use Yc, Xc only to estimate R; we return R and later apply it to *raw* Z.
    """
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R  # use later as Z_aligned = Z @ R  (no extra centering/scaling)

def _centroids(Z: np.ndarray, labels_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroids[K,2], label_order[K]) for stable class order."""
    uniq = np.unique(labels_str)
    cent = np.stack([Z[labels_str == u].mean(axis=0) for u in uniq], axis=0)
    return cent, uniq

def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, 1:4]

def _make_gif(figs, out_path, duration=0.8):
    frames = [_fig_to_rgb(f) for f in figs]
    with imageio.get_writer(out_path, mode="I", loop=0, duration=duration) as w:
        for fr in frames:
            w.append_data(fr)

import gc

@torch.no_grad()
def pca_evolution_plot(
    model: torch.nn.Module,
    base_weights: Dict,
    gold_weights: Dict,
    task_vector,
    dataset,
    split: str,
    alpha_range: List[float],
    device: torch.device,
    saving_dir:Path,
    *,
    normalize: bool = False,
    align_on: str = "centroids",   # "centroids" or "points"
    allow_reflection: bool = True,
):
    """
    Two-pass streaming:
      Pass 1: recompute per-alpha, align, update global bounds; free arrays immediately.
      Pass 2: recompute per-alpha, align, create Figure; keep figures in RAM and return them.
    No saving, no showing, no titles/hints.
    """

    # -------- dataloader (deterministic: ensure your dataset uses shuffle=False) --------
    if isinstance(dataset, Dataset):
        dataloader = _build_dataloader(dataset=256)
    elif split == "Test":
        dataloader = dataset.get_test_dataloader()
    elif split == "Train":
        dataloader = dataset.get_train_dataloader()
    elif split == "Heldout":
        dataloader = dataset.get_heldout_dataloader()
    else:
        raise ValueError(f"Invalid dataset split {split}")

    color_mapping = get_color_mappings(dataset.dataset_name)
    

    # -------- reference at alpha = 0 (anchor space) --------
    model.load_state_dict(base_weights, strict=False)
    feats0, labs0, _ = extract_features(model.get_feature_extractor(), dataloader, normalize=normalize, device=device)
    Z0 = _pca_coords(feats0, device)
    labels_str0 = _labels_to_names(labs0, dataset.get_class_names())
    del feats0, labs0
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if align_on == "points":
        X_anchor = Z0
        order0 = None
    else:
        X_anchor, order0 = _centroids(Z0, labels_str0)

    global_xmin = global_xmax = None
    global_ymin = global_ymax = None

    for alpha in alpha_range:
        model.load_state_dict(base_weights, strict=False)
        model.to(device)
        if alpha != 0.0:
            task_vector.apply_to(model, scaling_coef=alpha, strict=False)

        feats, labs, _ = extract_features(model.get_feature_extractor(), dataloader, normalize=normalize, device=device)
        Z = _pca_coords(feats, device)
        labels_str = _labels_to_names(labs, dataset.get_class_names())

        if align_on == "points":
            if Z.shape[0] != Z0.shape[0]:
                raise RuntimeError("Point-wise alignment requires identical sample counts/order across alphas.")
            R = _orthogonal_matrix(Z, X_anchor, allow_reflection=allow_reflection)
        else:
            Yc, order = _centroids(Z, labels_str)
            if not np.array_equal(order, order0):
                remap = {lab: i for i, lab in enumerate(order)}
                Yc = np.stack([Yc[remap[lab]] for lab in order0], axis=0)
            R = _orthogonal_matrix(Yc, X_anchor, allow_reflection=allow_reflection)

        Z_aligned = Z @ R
        xmn, xmx = Z_aligned[:, 0].min(), Z_aligned[:, 0].max()
        ymn, ymx = Z_aligned[:, 1].min(), Z_aligned[:, 1].max()
        global_xmin = xmn if global_xmin is None else min(global_xmin, xmn)
        global_xmax = xmx if global_xmax is None else max(global_xmax, xmx)
        global_ymin = ymn if global_ymin is None else min(global_ymin, ymn)
        global_ymax = ymx if global_ymax is None else max(global_ymax, ymx)

        # free ASAP
        del feats, labs, Z, labels_str, Z_aligned, R
        if align_on == "centroids":
            del Yc, order
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # include gold model in bounds
    if gold_weights:
        model.load_state_dict(gold_weights, strict=False)
        feats_g, labs_g, _ = extract_features(model.get_feature_extractor(), dataloader, normalize=normalize, device=device)
        Zg = _pca_coords(feats_g, device)
        labels_str_g = _labels_to_names(labs_g, dataset.get_class_names())
        if align_on == "points":
            Rg = _orthogonal_matrix(Zg, X_anchor, allow_reflection=allow_reflection)
        else:
            Yg, orderg = _centroids(Zg, labels_str_g)
            if not np.array_equal(orderg, order0):
                remap = {lab: i for i, lab in enumerate(orderg)}
                Yg = np.stack([Yg[remap[lab]] for lab in order0], axis=0)
            Rg = _orthogonal_matrix(Yg, X_anchor, allow_reflection=allow_reflection)
        Zg_aligned_tmp = Zg @ Rg
        global_xmin = min(global_xmin, Zg_aligned_tmp[:, 0].min())
        global_xmax = max(global_xmax, Zg_aligned_tmp[:, 0].max())
        global_ymin = min(global_ymin, Zg_aligned_tmp[:, 1].min())
        global_ymax = max(global_ymax, Zg_aligned_tmp[:, 1].max())
        # clean temp arrays from gold (we'll recompute in pass 2)
        del feats_g, labs_g, Zg, labels_str_g, Zg_aligned_tmp, Rg
        if align_on == "centroids":
            del Yg, orderg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # finalize global limits
    pad = 0.03
    xspan = global_xmax - global_xmin
    yspan = global_ymax - global_ymin
    xlim = (global_xmin - pad * xspan, global_xmax + pad * xspan)
    ylim = (global_ymin - pad * yspan, global_ymax + pad * yspan)


    figs_alpha: List[plt.Figure] = []

    for alpha in alpha_range:
        model.load_state_dict(base_weights, strict=False)
        if alpha != 0.0:
            task_vector.apply_to(model, scaling_coef=alpha, strict=False)

        feats, labs, _ = extract_features(model.get_feature_extractor(), dataloader, normalize=normalize, device=device)
        Z = _pca_coords(feats, device)
        labels_str = _labels_to_names(labs, dataset.get_class_names())

        if align_on == "points":
            R = _orthogonal_matrix(Z, X_anchor, allow_reflection=allow_reflection)
        else:
            Yc, order = _centroids(Z, labels_str)
            if not np.array_equal(order, order0):
                remap = {lab: i for i, lab in enumerate(order)}
                Yc = np.stack([Yc[remap[lab]] for lab in order0], axis=0)
            R = _orthogonal_matrix(Yc, X_anchor, allow_reflection=allow_reflection)

        Z_aligned = Z @ R
        fig, ax = datamapplot.create_plot(
            Z_aligned, labels_str,
            label_color_map=color_mapping,
            label_over_points=True,
            label_font_size=30,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        figs_alpha.append(fig)

        # free arrays; keep the figure
        del feats, labs, Z, labels_str, Z_aligned, R
        if align_on == "centroids":
            del Yc, order
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # gold figure
    fig_gold = None
    if gold_weights:
        model.load_state_dict(gold_weights, strict=False)
        feats_g, labs_g, _ = extract_features(model.get_feature_extractor(), dataloader, normalize=normalize, device=device)
        Zg = _pca_coords(feats_g, device)
        labels_str_g = _labels_to_names(labs_g, dataset.get_class_names())
        if align_on == "points":
            Rg = _orthogonal_matrix(Zg, X_anchor, allow_reflection=allow_reflection)
        else:
            Yg, orderg = _centroids(Zg, labels_str_g)
            if not np.array_equal(orderg, order0):
                remap = {lab: i for i, lab in enumerate(orderg)}
                Yg = np.stack([Yg[remap[lab]] for lab in order0], axis=0)
            Rg = _orthogonal_matrix(Yg, X_anchor, allow_reflection=allow_reflection)
        Zg_aligned = Zg @ Rg

        fig_gold, axg = datamapplot.create_plot(
            Zg_aligned, labels_str_g,
            label_color_map=color_mapping,
            label_over_points=True,
            label_font_size=30,
        )
        axg.set_xlim(*xlim)
        axg.set_ylim(*ylim)

        # clean arrays; keep figure
        del feats_g, labs_g, Zg, labels_str_g, Zg_aligned, Rg
        
        if align_on == "centroids":
            del Yg, orderg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
    saving_dir.mkdir(parents=True, exist_ok=True)
    with open(saving_dir / f"pca_alpha_{len(alpha_range)}_figs.pkl", "wb") as f:
        pickle.dump(figs_alpha, f) 
    if gold_weights:
        with open(saving_dir / "pca_gold_fig.pkl", "wb") as f:
            pickle.dump(fig_gold, f)

    return figs_alpha, fig_gold


def get_color_mappings(dataset_name:str):
    
    if dataset_name == None:
        return None
    elif dataset_name == 'MNIST':
            return {'0': "#f64a4a", '1': "#2e862a", '2': "#0095ff", '3': "#af009d", '4': "#192c65", '5': "#00fffb", '6': "#8246b7", '7': "#92b301", '8': "#fa8f1e", '9': "#0a8d63"}
    elif dataset_name == 'CIFAR10':
        
        return {'airplane': "#f64a4a", 'automobile': "#2e862a", 'bird': "#0095ff", 'cat': "#af009d", 'deer': "#192c65", 'dog': "#00fffb", 'frog': "#8246b7", 'horse': "#92b301", 'ship': "#fa8f1e", 'truck': "#0a8d63"}



def umap_plot(
    features: torch.Tensor = None,
    labels: torch.Tensor = None,
    feature_extractor:nn.Module = None,
    dataloader:DataLoader = None,
    device:torch.device = torch.device('cpu'),
    class_names:List[str] = None,
    normalize:bool = False,
    n_neighbors:int = 15,
    min_dist:float = 0.1,
    n_components:int = 2,
    random_state:float = 11.0,
):
    
    if not(features and labels):
        if feature_extractor and dataloader:
            features, labels = extract_features(feature_extractor, dataloader, normalize, device)
        else:
            raise RuntimeError('One of the pairs `features and labels` or `feature extractor and dataloade` must be provided to the function.')
    

    class_names = dict(enumerate(class_names))
    vectorized_converter = np.vectorize(lambda x: class_names[x])
    labels_str = vectorized_converter(labels)
        
    z = torchdr.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        device=device,
        backend="keops",
        # backend=None,
        random_state=random_state
    ).fit_transform(features)
    z = z.detach().cpu().numpy()
        
    fig, ax = datamapplot.create_plot(
        z,
        labels_str,
        label_over_points=True,
        label_font_size=30,
    )
    
    return fig
    
    
def pca_plot(
    features: torch.Tensor = None,
    labels: torch.Tensor = None,
    feature_extractor:nn.Module = None,
    dataloader:DataLoader = None,
    device:torch.device = torch.device('cpu'),
    class_names:List[str] = None,
    normalize:bool = False,
    n_components:int = 2,
    dataset_name:str = None,
    random_state:float = 11.0,
):
    
    if not(features and labels):
        if feature_extractor and dataloader:
            features, labels = extract_features(feature_extractor, dataloader, normalize, device)
        else:
            raise RuntimeError('One of the pairs `features and labels` or `feature extractor and dataloade` must be provided to the function.')
    

    class_names = dict(enumerate(class_names))
    vectorized_converter = np.vectorize(lambda x: class_names[x])
    labels_str = vectorized_converter(labels)
        
    z = torchdr.PCA(
        n_components=n_components,
        device=device,
        random_state=random_state
    ).fit_transform(features)
    z = z.detach().cpu().numpy()
        
    color_mapping = get_color_mappings(dataset_name)
    fig, ax = datamapplot.create_plot(
        z,
        labels_str,
        label_color_map=color_mapping,
        label_over_points=True,
        label_font_size=30,
    )
    
    # fig, ax = plt.subplots(figsize=(5, 5))  # you can change the size

    # ax.scatter(z[:, 0], z[:, 1], 
    #         c=labels.numpy(), cmap="tab10", 
    #         s=1, alpha=0.3)

    # # ax.set_title("PCA", fontsize=fontsize)
    # ax.set_xticks([-5, 5])
    # ax.set_yticks([-5, 5])
        
    return fig
    


def tsne_plot(
    features: torch.Tensor = None,
    labels: torch.Tensor = None,
    feature_extractor:nn.Module = None,
    dataloader:DataLoader = None,
    device:torch.device = torch.device('cpu'),
    class_names:List[str] = None,
    normalize:bool = False,
    n_components:int = 2,
    perplexity:int = 50,
    max_iter:int = 2000,
    random_state:float = 11.0,
):
    
    if not(features and labels):
        if feature_extractor and dataloader:
            features, labels = extract_features(feature_extractor, dataloader, normalize, device)
        else:
            raise RuntimeError('One of the pairs `features and labels` or `feature extractor and dataloade` must be provided to the function.')
    

    class_names = dict(enumerate(class_names))
    vectorized_converter = np.vectorize(lambda x: class_names[x])
    labels_str = vectorized_converter(labels)
        
    z = torchdr.TSNE(
        n_components=n_components,
        perplexity=perplexity,
        device=device,
        max_iter=max_iter,
        backend="keops",
        random_state=random_state
    ).fit_transform(features)
    z = z.detach().cpu().numpy()
        
    fig, ax = datamapplot.create_plot(
        z,
        labels_str,
        label_over_points=True,
        label_font_size=30,
    )
    
    return fig



def all_plot_comp(
    features: torch.Tensor = None,
    labels: torch.Tensor = None,
    feature_extractor:nn.Module = None,
    dataloader:DataLoader = None,
    device:torch.device = torch.device('cpu'),
    class_names:List[str] = None,
    normalize:bool = False,
    random_state:float = 11.0,
):
    
    if not(features and labels):
        if feature_extractor and dataloader:
            features, labels = extract_features(feature_extractor, dataloader, normalize, device)
        else:
            raise RuntimeError('One of the pairs `features and labels` or `feature extractor and dataloade` must be provided to the function.')
    

    class_names = dict(enumerate(class_names))
    vectorized_converter = np.vectorize(lambda x: class_names[x])
    labels_str = vectorized_converter(labels)
        
    umap = torchdr.UMAP(
        device=device,
        backend="keops",
        random_state=random_state
    ).fit_transform(features)
    z_umap = umap.detach().cpu().numpy()
        
    itsne = torchdr.InfoTSNE(
        device=device,
        backend="keops",
        random_state=random_state
    ).fit_transform(features)
    z_infotsne = itsne.detach().cpu().numpy()
    
    
    largevis = torchdr.LargeVis(
        device=device,
        backend="keops",
        random_state=random_state
    ).fit_transform(features)
    z_largevis = largevis.detach().cpu().numpy()
    
    pca = torchdr.PCA(
        device=device,
        backend="keops",
        random_state=random_state
    ).fit_transform(features)
    z_pca = pca.detach().cpu().numpy()
    
    # --- Plot the embeddings ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fontsize = 25

    scatter = axes[0].scatter(z_umap[:, 0], z_umap[:, 1], c=labels.numpy(), cmap="tab10", s=1, alpha=0.3)
    axes[0].set_title("UMAP", fontsize=fontsize)
    axes[0].set_xticks([-10, 10])
    axes[0].set_yticks([-10, 10])

    axes[1].scatter(z_infotsne[:, 0], z_infotsne[:, 1], c=labels.numpy(), cmap="tab10", s=1, alpha=0.3)
    axes[1].set_title("InfoTSNE", fontsize=fontsize)
    axes[1].set_xticks([-10, 10])
    axes[1].set_yticks([-10, 10])

    axes[2].scatter(z_largevis[:, 0], z_largevis[:, 1], c=labels.numpy(), cmap="tab10", s=1, alpha=0.3)
    axes[2].set_title("LargeVis", fontsize=fontsize)
    axes[2].set_xticks([-5, 5])
    axes[2].set_yticks([-5, 5])

    axes[3].scatter(z_pca[:, 0], z_pca[:, 1], c=labels.numpy(), cmap="tab10", s=1, alpha=0.3)
    axes[3].set_title("PCA", fontsize=fontsize)
    axes[3].set_xticks([-5, 5])
    axes[3].set_yticks([-5, 5])

    handles, lbls = scatter.legend_elements(prop="colors", size=15)
    legend_labels = [f"{i}" for i in range(10)]
    fig.legend(handles, legend_labels, loc="lower center", ncol=10, fontsize=15)
    plt.subplots_adjust(bottom=0.15, wspace=0.1)
        
    return fig




# def pca_evolution_plot(
#     model: nn.Module,
#     base_weights: Dict,
#     gold_weights: Dict,
#     task_vector,
#     dataset,
#     split: str, 
#     alpha_range: List[float],
#     device: torch.device,
#     saving_dir: Path,
# ):
    
#     def fig_to_rgb(fig):
#         """Return an (H, W, 3) uint8 array from a Matplotlib Figure for any backend."""
#         fig.canvas.draw()
#         w, h = fig.canvas.get_width_height()

#         # Try backends that support RGB directly (Agg, etc.)
#         try:
#             buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#             return buf.reshape(h, w, 3)
#         except AttributeError:
#             # TkAgg gives ARGB; convert to RGB
#             buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
#             # ARGB -> RGB by dropping alpha and reordering
#             return buf[:, :, 1:4]
    

#     def make_gif(figs, out_path="progress.gif", duration=0.8):
#         frames = [fig_to_rgb(f) for f in figs]
#         # Per-frame durations in seconds
#         with imageio.get_writer(out_path, mode="I", loop=0, duration=duration) as w:
#             for fr in frames:
#                 w.append_data(fr)
    
    
#     dataloader = None
#     if split == 'Test':
#         dataloader = dataset.get_test_dataloader()
#     elif split == 'Train':
#         dataloader = dataset.get_train_dataloader()
#     elif split == 'Heldout':
#         dataloader = dataset.get_heldout_dataloader()
#     else: raise ValueError(f'Invalid dataset split {split}')
    
#     figs_alpha = []
#     for alpha in alpha_range:
#         model.load_state_dict(base_weights, strict=False)
#         if alpha != 0.0:
#             task_vector.apply_to(model, scaling_coef=alpha, strict=False)
#         fig_pca = pca_plot(
#             feature_extractor=model.get_feature_extractor(),
#             dataloader=dataset.get_test_dataloader(),
#             class_names=dataset.get_class_names(),
#             dataset_name=dataset.dataset_name,
#             device=device,
#         )
        
#         figs_alpha.append(fig_pca)
    
#     with open(saving_dir / "pca_alpha_figs.pkl", "wb") as f:
#         pickle.dump(figs_alpha, f)
    
#     model.load_state_dict(gold_weights, strict=False)
#     fig_gold = pca_plot(
#         feature_extractor=model.get_feature_extractor(),
#         dataloader=dataloader,
#         class_names=dataset.get_class_names(),
#         dataset_name=dataset.dataset_name,
#         device=device,
#     )
    
#     with open(saving_dir / "pca_gold_fig.pkl", "wb") as f:
#         pickle.dump(figs_alpha, f)
    
#     # big_fig.savefig(results_dirs['embed_plots'] / "pca_evol_test.png", bbox_inches="tight")
    
#     # make_gif(figs_pca, results_dirs['embed_plots'] / "pca_evol_test.gif", duration=5.0)
    
#     # fig_pca_gold.savefig(results_dirs['embed_plots'] / "pca_gold.png", bbox_inches="tight")

#     return figs_alpha, fig_gold
