import comet_ml
from src.datasets import dataset_factory, CIFAR10, CIFAR100, BaseClassificationDataset, dataset_wrappers
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer, GradientAscentTrainer, utils as trainer_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import misc_utils
import torch

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
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis

import math
from typing import Sequence, Optional, Tuple, Union
import textwrap

def show_image_grid(
    images: Sequence[torch.Tensor],
    labels: Sequence[Union[str, int]],
    max_images: int = 64,
    figsize_per_cell: Tuple[float, float] = (2.2, 2.2),
    title: Optional[str] = None,
    label_wrap: int = 24,
    label_fontsize: int = 9,
    tight: bool = True,
):
    """
    Visualize (up to 64) images in a balanced grid with labels under each image.

    Args:
        images: Sequence of PyTorch tensors. Each can be HxW, CxHxW, or HxWxC.
                Channel count can be 1, 3, or 4 (alpha ignored).
        labels: Sequence of labels (same length as images). Will be str()'d.
        max_images: Maximum number of images to display (<=64).
        figsize_per_cell: (width,height) inches per grid cell.
        title: Optional figure title.
        label_wrap: Wrap labels to this many characters for readability.
        label_fontsize: Font size for labels under images.
        tight: If True, uses tight_layout() to minimize whitespace.

    Returns:
        (fig, axes): Matplotlib figure and axes array.
    """
    assert len(images) == len(labels), "images and labels must have same length."
    n = min(len(images), max_images, 64)
    if n == 0:
        raise ValueError("No images to display.")

    # --- Choose a near-square grid: cols ≈ rows ≈ sqrt(n)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # --- Figure size
    fig_w = max(1.0, cols * figsize_per_cell[0])
    fig_h = max(1.0, rows * figsize_per_cell[1] + 0.2)  # a little extra for labels
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.atleast_2d(axes).reshape(rows, cols)

    def _to_numpy_img(t: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
        """
        Convert a torch Tensor to a numpy image in HxW or HxWx3, float in [0,1].
        Returns (img, cmap) where cmap is 'gray' for single-channel, else None.
        """
        # Move to CPU and detach
        t = t.detach().cpu()

        # Accept: HxW, CxHxW, or HxWxC
        if t.ndim == 2:
            arr = t
            is_chw = False
        elif t.ndim == 3:
            C_first = t.shape[0] in (1, 3, 4)
            C_last = t.shape[-1] in (1, 3, 4)
            if C_first and not C_last:
                # CHW -> HWC
                t = t.permute(1, 2, 0)
            elif not C_first and not C_last:
                # Ambiguous; assume already HWC
                pass
            # Drop alpha if present
            if t.shape[-1] == 4:
                t = t[..., :3]
            arr = t
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}")

        arr = arr.numpy()

        # Determine if grayscale
        cmap = None
        if arr.ndim == 2:
            cmap = "gray"
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
            cmap = "gray"
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            cmap = None
        else:
            raise ValueError(f"Unsupported image shape after processing: {arr.shape}")

        # Convert to float32 in [0,1]
        if np.issubdtype(arr.dtype, np.floating):
            # If outside [0,1], min-max normalize to preserve contrast
            amin, amax = np.nanmin(arr), np.nanmax(arr)
            if not np.isfinite(amin) or not np.isfinite(amax):
                arr = np.nan_to_num(arr, nan=0.0)
                amin, amax = np.min(arr), np.max(arr)
            if amin < 0.0 or amax > 1.0:
                if amax > amin:
                    arr = (arr - amin) / (amax - amin)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
            arr = arr.astype(np.float32)
        else:
            # Integer types -> scale to [0,1]
            amax = np.max(arr)
            if amax == 0:
                arr = np.zeros_like(arr, dtype=np.float32)
            elif amax <= 255:
                arr = (arr / 255.0).astype(np.float32)
            else:
                # Fallback: min-max
                amin = np.min(arr)
                if amax > amin:
                    arr = ((arr - amin) / (amax - amin)).astype(np.float32)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)

        return arr, cmap

    # --- Plot images
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            img_np, cmap = _to_numpy_img(images[i])
            ax.imshow(img_np, cmap=cmap, interpolation="nearest")
            ax.set_axis_off()
            # Label beneath image (use title with padding)
            label_text = textwrap.fill(str(labels[i]), width=label_wrap)
            ax.set_title(label_text, fontsize=label_fontsize, pad=4)
        else:
            ax.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    if tight:
        plt.tight_layout(pad=0.5, rect=(0, 0, 1, 0.97) if title else None)

    return fig, axes



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
        tuple: (metrics_dict, all_predictions, all_targets, misclassified_indices)
    """
    loss_met = misc_utils.AverageMeter()
    model.reset_metrics()
    all_preds = []
    all_targets = []
    misclassified_indices = []
    
    if dataloader is None:
        return None, None, None, None
    
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
            misclassified_indices.extend(indices[mis_mask].cpu().tolist())
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return (
        metric_results,
        torch.tensor(all_preds),
        torch.tensor(all_targets),
        misclassified_indices,
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
    clean_set, noisy_set = dataset_cpy.get_clean_noisy_subsets(set='Train')
    
    dataset_cpy.set_trainset(clean_set, shuffle=False)
    clean_metric, _, _, misclassified_cleans = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    noisy_metric, _, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dummy_instance = noisy_set
    while not isinstance(dummy_instance, dataset_wrappers.NoisyClassificationDataset):
        dummy_instance = dummy_instance.dataset
    dummy_instance.switch_to_clean_lables()
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    healing_metric, _, _, misclassified_healed = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
        'noisy_set': noisy_metric,
        'healing_noise': healing_metric,
    }, misclassified_cleans, misclassified_healed
    
    
def apply(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
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
    
    
    dataset_cfg = cfg['datasets'][0]
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: dataset.get_class_names()} 
    model = model_factory.create_model(cfg['model'])
    model.freeze_all_heads()
    
    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
    dataset_cfg['train_transforms'] = model.get_val_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    original_dataset = None
    consistency_scores = None
    if dataset_cfg['name'] == 'cifar10':
        original_dataset = CIFAR10()
        consistency_scores = np.load('cifar10-cscores.npz')['scores']
        cs_lbls = np.load('cifar10-cscores.npz')['labels']
    elif dataset_cfg['name'] == 'cifar100':
        original_dataset = CIFAR100()
        consistency_scores = np.load('cifar100-cscores.npz')['scores']
        cs_lbls = np.load('cifar100-cscores.npz')['labels']
    original_dataset.reset_train_dl(shuffle=False)
    
    train_indices = np.array(dataset.get_train_indices())
    
    heldout_indices = np.array(dataset.get_heldout_indices())
    
    consistency_scores = consistency_scores[train_indices]
    cs_lbls = cs_lbls[train_indices]
    
    # imgs = []
    # lbls = []
    
    # k = 32
    # for idx in np.argsort(consistency_scores)[:k]:
    #     img, lbl, dx = original_dataset.get_trainset()[idx]
    #     imgs.append(img)
    #     lbls.append(lbl)
    
    # class_names = dict(enumerate(original_dataset.get_class_names()))
    # vectorized_converter = np.vectorize(lambda x: class_names[x])
    # labels_str = vectorized_converter(lbls)
    
    # fig, _ = show_image_grid(
    #     images=imgs,
    #     labels=labels_str,
    #     max_images=32,
    # )
    # plt.show()
    
    # print(consistency_scores[np.argsort(consistency_scores)[-k:][::-1]])
    # print(consistency_scores[np.argsort(consistency_scores)[:k]])
    # exit()
    
    dataset.reset_train_dl(shuffle=False)
    
    dataset_clean = copy.deepcopy(dataset)
    
    strategy = cfg['strategy']
    dataset.inject_noise(**strategy['noise']['pretraining'])



    # Load weights while removing classifier weights from the state dict
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    gold_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_ho_clean_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"finetune_clean/weights/ft_weights.pth"),
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
        task_vectors['Average TV'] = only_tv
    else:
        task_vectors['Average TV'] = TaskVector.mean(task_vectors)
        
    
    task_vectors['Clean'] = TaskVector(mix_weights, ft_ho_clean_weights)
    


    model.load_state_dict(mix_weights, strict=False)
    
    task_vectors['Average TV'].apply_to(model, scaling_coef=-0.5, strict=False)
    train_results, misclassified_cleans, misclassified_heals = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    print(train_results)
    print(misclassified_cleans)
    print(consistency_scores[misclassified_cleans])
    


    
    # with open(results_dir / 'tv_metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)
    
    




if __name__ == "__main__":
    

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")

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

    if cfg['datasets'][0]['name'] != 'cifar10' and cfg['datasets'][0]['name'] != 'cifar100':
        raise ValueError(f'Consistency scores for {cfg['datasets'][0]['name']} are not available.')
    apply(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)