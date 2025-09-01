import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchdr
from typing import List
import datamapplot

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
        
    fig, ax = datamapplot.create_plot(
        z,
        labels_str,
        label_over_points=True,
        label_font_size=30,
    )
    
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


# # Run UMAP
# def umap_plot(
#     feature_extractor:nn.Module,
#     dataloader:DataLoader,
#     device:torch.device = torch.device('cpu'),
#     normalize:bool = False,
#     n_neighbors:int = 15,
#     min_dist:float = 0.1,
#     n_components:int = 2
# ):
#     features, labels = extract_features(feature_extractor, dataloader, normalize, device)
    
#     reducer = umap.UMAP(n_neighbors=n_neighbors,
#                         min_dist=min_dist,
#                         n_components=n_components,
#                         random_state=42)
#     embedding = reducer.fit_transform(features)
    
#     plt.figure(figsize=(8,6))
#     scatter = plt.scatter(embedding[:,0], embedding[:,1], 
#                           c=labels, cmap="tab10", s=10, alpha=0.7)
#     plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.title("UMAP of Image Encoder Features")
#     plt.show()

