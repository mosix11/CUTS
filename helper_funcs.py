import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from src.datasets import dataset_factory, dataset_wrappers, BaseClassificationDataset
from src.models import model_factory, TaskVector
from src.utils import nn_utils, misc_utils
import copy
from tqdm import tqdm
from torchmetrics.classification import MulticlassConfusionMatrix
from typing import Union

from torch.optim.swa_utils import update_bn

def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch





def recalibrate_batchnorm(model:torch.nn.Module, dataloader:DataLoader, device:torch.device, max_batches=None):
    # Recompute BN running_mean / running_var using data from `loader`.
    # This does *not* change weights; it only updates BN buffers.
    model.to(device)
    update_bn(dataloader, model, device=device)


def evaluate_model(
    model:torch.nn.Module,
    dataloader:DataLoader,
    device:torch.device
):
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: A tuple containing (all_predictions, all_targets, metrics_dict).
    """
    loss_met = misc_utils.AverageMeter()
    model.reset_metrics()
    all_preds = []
    all_targets = []
    
    if dataloader == None:
        return None, None, None
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            input_batch, target_batch = batch[:2]
            
            loss, preds = model.validation_step(input_batch, target_batch, use_amp=True, return_preds=True)
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            predictions = torch.argmax(preds, dim=-1)
            all_preds.extend(predictions.cpu())
            all_targets.extend(target_batch.cpu())
            
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return metric_results, torch.tensor(all_preds), torch.tensor(all_targets) 


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
    clean_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    noisy_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dummy_instance = noisy_set
    while not isinstance(dummy_instance, (dataset_wrappers.NoisyClassificationDataset, dataset_wrappers.PoisonedClassificationDataset)):
        dummy_instance = dummy_instance.dataset
    dummy_instance.switch_to_clean_lables()
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    healing_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
        'noisy_set': noisy_metric,
        'healing_noise': healing_metric,
    }

    
def search_optimal_coefficient(
    base_model:torch.nn.Module,
    task_vector:TaskVector,
    search_range:tuple,
    dataset:BaseClassificationDataset,
    num_classes:int,
    device:torch.device
):
    """
    Performs a search to find the optimal task vector scaling coefficient.

    Args:
        base_model (torch.nn.Module): The pre-trained model. A deepcopy is made for each evaluation.
        task_vector (TaskVector): The task vector object.
        dataset: The dataset object to get the validation and test dataloaders from.
        search_range (list or tuple): A list/tuple [min_val, max_val] for the search.
        device (torch.device): The device to run evaluation on.
        num_classes (int): The number of classes for the confusion matrix.

    Returns:
        tuple: (best_coefficient, best_performance_metrics, confusion_matrix_tensor)
    """
    
    # If the dataset has a validation set, we optimize on the validation set,
    # if the dataset doesn't have a validation set, we optimize on the heldout set,
    # and if the dataset does not have validation or heldout set, we optimize on the
    # test set.
    val_dataloader = dataset.get_val_dataloader()
    if val_dataloader == None or len(val_dataloader) == 0:
        val_dataloader = dataset.get_heldout_dataloader()
        if val_dataloader == None or len(val_dataloader) == 0:
            val_dataloader = dataset.get_test_dataloader()
            print("Optimizing on the test set.")
        else:
            print("Optimizing on the heldout set.")
    else:
        print("Optimizing on the validation set.")
    test_dataloader = dataset.get_test_dataloader()
    
    best_coef = 0.0
    best_acc = -1.0
    best_results = {}
    
    print("--- Starting Coarse Search ---")
    coarse_search_grid = np.arange(search_range[0], search_range[1] + 0.1, 0.1)
    
    for scale_coef in tqdm(coarse_search_grid, desc="Coarse Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, val_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results
    
    # print(f"\nCoarse search best coefficient: {best_coef:.2f} with Accuracy: {best_acc:.4f}")

    print("\n--- Starting Fine Search ---")
    fine_search_start = max(search_range[0], best_coef - 0.1)
    fine_search_end = min(search_range[1], best_coef + 0.1)
    fine_search_grid = np.linspace(fine_search_start, fine_search_end, num=21)

    for scale_coef in tqdm(fine_search_grid, desc="Fine Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, val_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results

    # print(f"\nRecalculating metrics and confusion matrix for best coefficient: {best_coef:.2f}")
    final_model = copy.deepcopy(base_model)
    task_vector.apply_to(final_model, scaling_coef=best_coef)
    final_model.to(device)

    best_results, all_preds, all_targets = evaluate_model(final_model, test_dataloader, device)
    
    confmat_metric = MulticlassConfusionMatrix(num_classes=num_classes)
    best_cm_tensor = confmat_metric(all_preds, all_targets)

    return best_coef, best_results, best_cm_tensor




def get_confusion_matrix(
    model:torch.nn.Module,
    num_classes:int,
    dataloader:DataLoader,
    device:torch.device,
    normalize:bool=False
):
    model.eval()
    model.to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    
    for i, batch in enumerate(dataloader):
        batch = batch[:2]
        batch = prepare_batch(batch, device)
        input_batch, target_batch = batch
        
        model_output = model.predict(input_batch) # Get raw model output (logits)
        predictions = torch.argmax(model_output, dim=-1) # Get predicted class labels
        
        cm_metric.update(predictions.detach(), target_batch.detach())
    
    cm = cm_metric.compute().cpu().numpy()
    if normalize:
        cm = row_normalize(cm)
    return cm
    
    

def row_normalize(cm):
    with np.errstate(divide='ignore', invalid='ignore'):
        rn = cm / cm.sum(axis=1, keepdims=True)
        rn[np.isnan(rn)] = 0.0
    return rn
    
def top_confused_pairs(cm, class_names=None, k=10):
    # exclude diagonal (correct)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(off.shape[0]):
        for j in range(off.shape[1]):
            if off[i, j] > 0:
                pairs.append((i, j, off[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    def name(ix): 
        return class_names[ix] if class_names else ix
    return [(name(i), name(j), n) for i, j, n in pairs[:k]]

def asymmetry_scores(cm, threshold=5, class_names=None):
    """
    For each (i,j), compute A_ij = (cm[i,j]-cm[j,i]) / (cm[i,j]+cm[j,i])
    Only report where total confusions >= threshold.
    """
    C = cm.copy().astype(float)
    out = []
    K = C.shape[0]
    for i in range(K):
        for j in range(i+1, K):
            a, b = C[i, j], C[j, i]
            total = a + b
            if total >= threshold:
                A_ij = (a - b) / total
                pair = (i, j, A_ij, int(a), int(b))
                out.append(pair)
    # sort by absolute asymmetry
    out.sort(key=lambda x: abs(x[2]), reverse=True)
    def name(ix): 
        return class_names[ix] if class_names else ix
    return [ (name(i), name(j), aij, a, b) for i, j, aij, a, b in out ]

def per_class_confusion_entropy(cm):
    """
    Entropy of row-normalized cm per class (higher => more diffuse confusions).
    """
    rn = row_normalize(cm)
    eps = 1e-12
    H = -np.sum(rn * np.log(rn + eps), axis=1)
    return H



def analyze_IC(
    model:torch.nn.Module,
    num_classes:int,
    dataloader:DataLoader,
    device:torch.device,
    class_names=None
):
    cm = get_confusion_matrix(
        model,
        num_classes,
        dataloader,
        device
    )
    rn = row_normalize(cm)

    print("\nTop confused pairs (true → predicted):")
    for t, p, n in top_confused_pairs(cm, class_names, k=10):
        print(f"{t} → {p}: {n}")

    print("\nMost asymmetric pairs (A_ij near ±1 means strongly one-directional):")
    for i, j, aij, a, b in asymmetry_scores(cm, threshold=5, class_names=class_names)[:10]:
        print(f"{i} ↔ {j}: A={aij:+.3f}  ({i}→{j}={a}, {j}→{i}={b})")

    H = per_class_confusion_entropy(cm)
    if class_names:
        print("\nPer-class confusion entropy:")
        for name, h in sorted(zip(class_names, H), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{name}: {h:.3f}")
    return cm, rn



def estimate_T_from_confusion(cm, alpha=0.5, lam=0.1):
    """
    cm: (K,K) confusion counts on a CLEAN set (true rows, predicted cols)
    alpha: Laplace smoothing for off-diagonal
    lam: shrinkage toward uniform off-diagonal
    returns: T (K,K), zero diagonal, rows sum to 1
    """
    K = cm.shape[0]
    T = np.zeros_like(cm, dtype=float)

    # Off-diagonal smoothing
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0.0)
    off += alpha  # Laplace on off-diagonal cells

    row_sums_off = off.sum(axis=1, keepdims=True)
    # If a row has zero mistakes, fall back to uniform off-diagonal
    uniform_off = np.full((K, K), 1.0/(K-1))
    np.fill_diagonal(uniform_off, 0.0)

    # Row-normalize off-diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        T_hat = np.divide(off, row_sums_off, where=row_sums_off > 0)
    # For rows with no mistakes: use uniform off-diagonal
    zero_rows = (row_sums_off.flatten() == 0)
    if zero_rows.any():
        T_hat[zero_rows] = uniform_off[zero_rows]

    # Shrinkage toward uniform
    T = (1 - lam) * T_hat + lam * uniform_off
    # Ensure exact zeros on diagonal and row sums = 1
    np.fill_diagonal(T, 0.0)
    T /= T.sum(axis=1, keepdims=True)
    return T

def rowwise_kl_to_uniform(T):
    K = T.shape[0]
    U = np.full_like(T, 1.0/(K-1))
    np.fill_diagonal(U, 0.0)
    # avoid log(0)
    eps = 1e-12
    M = T + eps
    return np.sum(M * (np.log(M) - np.log(U + eps)), axis=1)

def symmetric_noise_detected(T, kl_thresh=0.03):
    kl = rowwise_kl_to_uniform(T)
    return bool((kl < kl_thresh).all()), kl