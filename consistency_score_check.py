import os
PYTHON_HASH_SEED = 0
os.environ["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import comet_ml
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
torch.set_float32_matmul_precision("high")

from src.datasets import dataset_factory, CIFAR10, CIFAR100,MNIST, BaseClassificationDataset, dataset_wrappers
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer, GradientAscentTrainer, utils as trainer_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import misc_utils

import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, Subset, ConcatDataset
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv
import yaml
import pickle
import copy
import random
import numpy as np
from torchmetrics import ConfusionMatrix
import json
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import re

import imageio.v2 as imageio

from src.utils import embedding_space_analysis
from helper_funcs import get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis

import math
from typing import Sequence, Optional, Tuple, Union
import textwrap

import math
from typing import Sequence, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def show_image_grid(
    images: Sequence[torch.Tensor],
    labels: Sequence[Union[str, int]],
    max_images: int = 64,
    n_rows: int = 8,
    n_cols: int = 8,
    figsize_per_cell: Tuple[float, float] = (2.2, 2.6),
    title: Optional[str] = None,
    label_wrap: int = 24,
    label_fontsize: int = 9,
    image_height_frac: float = 0.88,   # fraction of the cell height used by the image
    label_band_frac: float = 0.1,     # explicit label band height (leave a bit of gap below)
    hspace: float = 0.25,              # vertical spacing between grid rows
    wspace: float = 0.08,              # horizontal spacing between grid cols
):
    """
    Visualize (up to 64) images in a balanced grid with a dedicated label band per cell.
    Images are drawn in an inset occupying the top portion of the cell; labels sit below.

    Args are similar to before, with the key layout params:
      - image_height_frac: portion of the cell reserved for the image (0..1)
      - label_band_frac: portion reserved for the label text (0..1)
      - hspace/wspace: inter-cell spacing (as in GridSpec semantics)
    """
    assert len(images) == len(labels), "images and labels must have same length."
    n = min(len(images), max_images, 64)
    if n == 0:
        raise ValueError("No images to display.")

    # --- near-square grid
    if n_cols: cols = n_cols
    else: cols = math.ceil(math.sqrt(n))
    if n_rows: rows = n_rows
    else: rows = math.ceil(n / cols)

    # --- figure size
    fig_w = max(1.0, cols * figsize_per_cell[0])
    fig_h = max(1.0, rows * figsize_per_cell[1])
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(rows, cols, hspace=hspace, wspace=wspace)

    # sanity for fractions
    image_height_frac = float(np.clip(image_height_frac, 0.5, 0.95))
    label_band_frac = float(np.clip(label_band_frac, 0.05, 0.45))

    def _to_numpy_img(t: torch.Tensor):
        """Return (np_img, cmap). Accepts HxW, CxHxW, or HxWxC; normalizes to [0,1]."""
        t = t.detach().cpu()
        if t.ndim == 2:
            arr = t
        elif t.ndim == 3:
            if t.shape[0] in (1, 3, 4):  # CHW -> HWC
                t = t.permute(1, 2, 0)
            if t.shape[-1] == 4:
                t = t[..., :3]
            arr = t
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}")

        arr = arr.numpy()
        cmap = None
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
            cmap = "gray"
        elif arr.ndim == 2:
            cmap = "gray"

        # normalize to [0,1] robustly
        arr = arr.astype(np.float32, copy=False)
        amin = float(np.nanmin(arr))
        amax = float(np.nanmax(arr))
        if not np.isfinite(amin) or not np.isfinite(amax):
            arr = np.nan_to_num(arr, nan=0.0)
            amin, amax = float(np.min(arr)), float(np.max(arr))
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        return arr, cmap

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        cell_ax = fig.add_subplot(gs[r, c])
        cell_ax.set_xticks([]); cell_ax.set_yticks([])
        for spine in cell_ax.spines.values():
            spine.set_visible(False)

        if i >= n:
            cell_ax.set_visible(False)
            continue

        img_np, cmap = _to_numpy_img(images[i])

        # --- draw image in an inset occupying the top portion of the cell
        top = 1.0
        img_h = image_height_frac
        img_ax = cell_ax.inset_axes([0.0, top - img_h, 1.0, img_h])  # [x0, y0, w, h] in cell axes coords
        img_ax.imshow(img_np, cmap=cmap, interpolation="nearest")
        img_ax.set_axis_off()

        # --- label text in the reserved band below
        label_text = textwrap.fill(str(labels[i]), width=label_wrap)
        # Place text centered near the bottom of the cell (inside padding band)
        cell_ax.text(
            0.5, label_band_frac * 0.6,  # vertical anchor inside label band
            label_text,
            ha="center", va="center",
            fontsize=label_fontsize,
            transform=cell_ax.transAxes,
            wrap=True,
        )

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)

    return fig

def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (metrics_dict, all_predictions, all_targets, misclassified_indices, misclassified_samples)
            - metrics_dict: evaluation metrics including loss
            - all_predictions: tensor of all predictions
            - all_targets: tensor of all targets
            - misclassified_indices: list of misclassified sample indices
            - misclassified_samples: list of (target, prediction) tuples for misclassified samples
    """
    loss_met = misc_utils.AverageMeter()
    model.reset_metrics()
    all_preds = []
    all_targets = []
    misclassified_indices = []
    misclassified_samples = []
    
    if dataloader is None:
        return None, None, None, None, None
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            input_batch, target_batch, indices = batch[:3]
            
            loss, preds = model.validation_step(
                input_batch, target_batch, use_amp=True, return_preds=True
            )
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            predictions = torch.argmax(preds, dim=-1)
            all_preds.extend(predictions.cpu())
            all_targets.extend(target_batch.cpu())
            
            # find misclassified
            mis_mask = predictions != target_batch
            mis_indices = indices[mis_mask].cpu().tolist()
            mis_targets = target_batch[mis_mask].cpu().tolist()
            mis_preds = predictions[mis_mask].cpu().tolist()
            
            misclassified_indices.extend(mis_indices)
            misclassified_samples.extend(list(zip(mis_targets, mis_preds)))
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return (
        metric_results,
        torch.tensor(all_preds),
        torch.tensor(all_targets),
        misclassified_indices,
        misclassified_samples,
    )

def eval_model_on_clean_noise_splits(
    model:torch.nn.Module,
    cfg:dict,
    dataset:BaseClassificationDataset,
    device:torch.device
):
    dataset_cpy = copy.deepcopy(dataset)
    if cfg is not None:
        strategy = cfg['strategy']
        dataset_cpy.inject_noise(**strategy['noise']['pretraining'])
        
    try:
        clean_set, noisy_set = dataset_cpy.get_clean_noisy_subsets(set='Train')
    except:
        clean_set, noisy_set = dataset.get_trainset(), None
    
    dataset_cpy.set_trainset(clean_set, shuffle=False)
    clean_metric, _, _, misclassified_cleans, misclassified_cleans_smp = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    
    if noisy_set:
        dataset_cpy.set_trainset(noisy_set, shuffle=False)
        noisy_metric, _, _, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
        
        dummy_instance = noisy_set
        while not isinstance(dummy_instance, dataset_wrappers.NoisyClassificationDataset):
            dummy_instance = dummy_instance.dataset
        dummy_instance.switch_to_clean_lables()
        
        dataset_cpy.set_trainset(noisy_set, shuffle=False)
        healing_metric, _, _, misclassified_healed, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
    }, misclassified_cleans, misclassified_cleans_smp, _
    
    # return {
    #     'clean_set': clean_metric,
    #     'noisy_set': noisy_metric,
    #     'healing_noise': healing_metric,
    # }, misclassified_cleans, misclassified_cleans_smp, misclassified_healed
    
    
def apply(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    dataset_seed = cfg['dataset_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    outputs_dir = outputs_dir / cfg_name
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_dirs = {}
    results_dirs['metrics'] = results_dir / 'metrics'
    for dir in results_dirs.values():
        dir.mkdir(exist_ok=True, parents=True)
    
    
    try:
        dataset_cfg = cfg['datasets'][0]
    except:
        dataset_cfg = cfg['dataset']
        
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: dataset.get_class_names()} 
    model = model_factory.create_model(cfg['model'])
    model.freeze_all_heads()
    
    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
    dataset_cfg['train_transforms'] = model.get_val_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    dataset.reset_train_dl(shuffle=False)
    
    dataset_clean = copy.deepcopy(dataset)
    
    strategy = cfg['strategy']
    if 'pretraining' in strategy['noise']:
        dataset.inject_noise(**strategy['noise']['pretraining'])
    
    try:
        train_indices = np.array(dataset.get_train_indices())
    except:
        train_indices = np.arange(len(dataset.get_trainset()))

    
    # heldout_indices = np.array(dataset.get_heldout_indices())
    
    # consistency_scores = consistency_scores[train_indices]
    # cs_lbls = cs_lbls[train_indices]
    
    original_dataset = None
    consistency_scores = None
    if dataset_cfg['name'] == 'cifar10':
        original_dataset = CIFAR10()
        consistency_scores = np.load('./consistency_scores/cifar10-cscores.npz')['scores']
        cs_lbls = np.load('./consistency_scores/cifar10-cscores.npz')['labels']
    elif dataset_cfg['name'] == 'cifar100':
        original_dataset = CIFAR100()
        consistency_scores = np.load('./consistency_scores/cifar100-cscores.npz')['scores']
        cs_lbls = np.load('./consistency_scores/cifar100-cscores.npz')['labels']
    elif dataset_cfg['name'] == 'mnist':
        original_dataset = MNIST()
        # TODO find the cons score for MNIST with matching samples indices with torch version.
        # consistency_scores = np.load('consistency_scorse/cifar100-cscores.npz')['scores']
        # cs_lbls = np.load('cifar100-cscores.npz')['labels']
    original_dataset.reset_train_dl(shuffle=False)
    
    
    #     # alpha -1.35
    # cifar10_cfg30_clean_forgotten_indices = {
    #     5019: (5, 3),
    #     11246: (5, 2),
    #     23870: (0, 8),
    #     31937: (8, 2),
    #     32367: (5, 3),
    #     32426: (5, 7),
    #     36395: (3, 6),
    #     36419: (7, 4)
    # }
    
    # # alpha -0.8
    # cifar100_cfg31_clean_forgotten_indices = {
    #     9981: (42, 88),
    #     15980: (60, 71),
    #     19370: (11, 2),
    #     25514: (88, 44),
    #     25318: (98, 46),
    #     31295: (51, 46),
    #     38595: (92, 62),
    #     44764: (33, 51),
    # }
    
    # # alpha -0.8
    # mnist_cfg37_clean_forgotten_indices = {
    #     52341: (9, 4),
    #     10777: (3, 9),
    #     26057: (7, 1),
    #     29483: (9, 5),
    #     34638: (5, 6),
    #     39366: (5, 3),
    #     42606: (5, 3),
    #     58724: (4, 7),
    # }
    
    # imgs = []
    
    # print(train_indices[list(cifar100_cfg31_clean_forgotten_indices.keys())])
    # exit()
    # for idx in train_indices[list(mnist_cfg37_clean_forgotten_indices.keys())]:
    #     img, lbl, dx = original_dataset.get_trainset()[idx]
    #     imgs.append(img)
    #     # lbls.append(lbl)
    
    
    # class_names = dict(enumerate(original_dataset.get_class_names()))
    # vectorized_converter = np.vectorize(lambda x: class_names[x])
    # # labels_str = vectorized_converter(lbls)
    # misclassified_strs = [
    #     fr"{vectorized_converter(t)} $\rightarrow$ {vectorized_converter(p)}"
    #     for (t, p) in list(mnist_cfg37_clean_forgotten_indices.values())
    # ]
        
    # fig = show_image_grid(
    #     images=imgs,
    #     labels=misclassified_strs,
    #     n_cols=2,
    #     n_rows=4,
    #     label_fontsize=12,
    #     label_wrap=36,
    #     hspace=0.05,
    #     wspace=0.01
    #     # max_images=32,
    # )
    # fig.savefig(fname=Path('./visulaization_dir')/ 'mnist_cfg37_clean_forgotten.png', bbox_inches="tight", dpi=300)
    # plt.show()
    

    # exit()
    # Load weights while removing classifier weights from the state dict
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    

    noise_weights = OrderedDict()
    
    for noise_tv in cfg['strategy']['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        noise_weights[f"{noise_tv['noise_rate']*100:.0f}% Noise, {noise_tv['seed']} Seed"] = n_weights
        
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average'] = only_tv
    else:
        task_vectors['Average'] = TaskVector.mean(task_vectors)
        

    model.load_state_dict(mix_weights, strict=False)
    
    task_vectors['Average'].apply_to(model, scaling_coef=-0.26, strict=False)
    train_results, misclassified_cleans, misclassified_cleans_smp, misclassified_heals = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    print(train_results)

    
    misclassified_mapping = dict(zip(misclassified_cleans, misclassified_cleans_smp))

    for idx, (key, value) in enumerate(misclassified_mapping.items()):
        print(idx+1, key, value)
    # print(consistency_scores[misclassified_cleans])
    
    imgs = []
    
    for idx in train_indices[misclassified_cleans]:
        img, lbl, dx = original_dataset.get_trainset()[idx]
        imgs.append(img)
        # lbls.append(lbl)
    
    
    class_names = dict(enumerate(original_dataset.get_class_names()))
    vectorized_converter = np.vectorize(lambda x: class_names[x])
    # labels_str = vectorized_converter(lbls)
    misclassified_strs = [
        fr"{vectorized_converter(t)} $\rightarrow$ {vectorized_converter(p)}"
        for (t, p) in misclassified_cleans_smp
    ]
        
    if len(imgs) > 64:
        imgs = imgs[-64:]
        misclassified_strs = misclassified_strs[-64:]
        
    fig = show_image_grid(
        images=imgs,
        labels=misclassified_strs,
        label_fontsize=12,
        label_wrap=36,
        hspace=0.05,
        wspace=0.01
        # max_images=32,
    )
    plt.show()
    





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_noise_TA").absolute()
    results_dir = Path("results/single_experiment/clip_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    # if cfg['datasets'][0]['name'] != 'cifar10' and cfg['datasets'][0]['name'] != 'cifar100':
    #     raise ValueError(f'Consistency scores for {cfg['datasets'][0]['name']} are not available.')
    apply(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)