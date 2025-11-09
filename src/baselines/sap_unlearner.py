import os
import torch
torch.backends.cuda.preferred_linalg_library("magma")
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import  Counter
from collections import OrderedDict
from tqdm import tqdm
import copy
from collections import defaultdict
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.utils import nn_utils, misc_utils

from torch.utils.data import Dataset, Subset, DataLoader
def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


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


def get_SVD (mat_dict, set_name = "SVD"):
    feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    s_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in mat_dict.keys():
        for act in tqdm(mat_dict[loc].keys(), desc=f"{loc}layer - SVD for {set_name}"):
            activation = torch.Tensor(mat_dict[loc][act]).to("cuda")
            if torch.isnan(activation).any() or torch.isinf(activation).any():
                raise ValueError("activation contains NaN or Inf values")
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            # U, S, Vh = torch.linalg.svd(activation.cpu(), full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()            
            feature_dict[loc][act] = U
            s_dict[loc][act] = S
    return feature_dict,  s_dict


def select_basis(feature_dict, full_s_dict, threshold):
    if threshold is None:
        return feature_dict
    out_feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = feature_dict[loc][act]
            S = full_s_dict[loc][act]
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold) +1  
            out_feature_dict[loc][act] = U[:,:r]
    print('-'*40)
    print(f'Gradient Constraints Summary')
    print('-'*40)
    for loc in out_feature_dict.keys():
        for act in out_feature_dict[loc].keys():
            print (f'{loc} layer {act} : {out_feature_dict[loc][act].shape[1]}/{out_feature_dict[loc][act].shape[0]}')
    print('-'*40)
    return out_feature_dict


     
def get_scaled_feature_mat(feature_dict, full_s_dict, alpha, device):
    feature_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    # Projection Matrix Precomputation
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = torch.Tensor( feature_dict[loc][act] ).to(device)
            S = full_s_dict[loc][act]
            r = U.shape[1]
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            importance =  torch.Tensor(( alpha*sval_ratio/((alpha-1)*sval_ratio+1) ) [:r]).to(device) 
            U.requires_grad = False
            feature_mat_dict[loc][act] = torch.mm( U, torch.diag(importance**0.5) )
    return feature_mat_dict



def get_projections(feature_mat_retain_dict, device):
    feature_mat = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_mat_retain_dict.keys():
        for act in feature_mat_retain_dict[loc].keys():
            Ur = feature_mat_retain_dict[loc][act]
            Mr = torch.mm(Ur, Ur.transpose(0,1))     
            # Select type of projection. 
            feature_mat[loc][act]= Mr

    return feature_mat




def get_representation_matrix(net, device, data_loader, set_name="Heldout Set"):
    """
    Collects per-layer representation matrices R_l for a clean held-out dataset.

    Args:
        net: model with .get_activations(batch) method
        device: torch device
        data_loader: DataLoader containing trusted samples
        set_name: just for printing/debugging
    Returns:
        mat_dict: dict["pre"|"post"] -> dict[layer_name] = activation_matrix (d x n)
                  where d is feature dim and n is number of collected samples/patches
    """
    net.eval()
    activations = None

    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Extracting representations for {set_name}")):
        images = batch[0]
        images = images.to(device)

        # Forward pass with hooks or model’s get_activations() method

        batch_activations = net.get_activations(images)


        # Concatenate across batches
        if activations is None:
            activations = {loc: {k: v.copy() for k, v in batch_activations[loc].items()} for loc in batch_activations.keys()}
        else:
            for loc in batch_activations.keys():
                for key in batch_activations[loc].keys():
                    activations[loc][key] = np.concatenate([activations[loc][key], batch_activations[loc][key]], axis=0)


    # Transpose: make columns = samples, rows = feature dimensions
    mat_dict = {loc: OrderedDict() for loc in activations.keys()}
    for loc in activations.keys():
        for key, act in activations[loc].items():
            mat_dict[loc][key] = act.T  # (dim, n_samples)

    # Print summary
    print("-" * 40)
    print(f"Representation Matrices for {set_name}")
    print("-" * 40)
    for loc in mat_dict.keys():
        for layer_name, mat in mat_dict[loc].items():
            print(f"  {loc} | {layer_name}: {mat.shape}")
    print("-" * 40)

    return mat_dict



def SAP_unlearning_noise(
        model,
        clean_samples_dl,
        test_dl,
        device,
        project_classifier_head = True,
        scale_coff_list = [1000, 5000, 10000, 30000, 50000, 100000, 300000, 500000, 1000000],
        ):
    """
    Applies the SAP algorithm using a clean held-out dataset to compute activation projections.

    Args:
        model: trained (possibly noisy) model
        clean_samples_dl: DataLoader of clean / trusted samples
        device: torch device
        scale_coff_list: list of alpha values for SAP scaling
        prev_recur_proj_mat: optional previous round projectors (for recurrent SAP)
    Returns:
        projected_models: dict { alpha : model_copy_with_projection }
        proj_mats: dict { alpha : projector_matrices_per_layer }
    """

    model.to(device)
    model.eval()
    
    best_alpha = None
    best_model = None
    best_ACC = -np.inf

    # Step 1: Collect activation representations on the clean dataset
    print("\nCollecting activation representations on clean set...")
    mat_retain_dict = get_representation_matrix(
        model, device, clean_samples_dl, set_name="Clean Set"
    )

    # Step 2: Compute SVDs (U, S) per layer
    print("\nComputing SVDs for each layer...")
    full_feature_retain_dict, full_s_retain_dict = get_SVD(
        mat_retain_dict, set_name="SVD Clean Set"
    )

    results_dict = OrderedDict()
    # Step 3: Loop over alphas to build projectors and update weights
    print("\nRunning SAP projection...")
    for alpha in scale_coff_list:
        print(f"  α = {alpha}")

        # Build scaled feature matrices (U Λ^{1/2})
        scaled_feature_dict = get_scaled_feature_mat(
            full_feature_retain_dict, full_s_retain_dict,
            alpha=alpha, device=device
        )

        # Compute projection matrices Mr = U Λ Uᵀ (Eq.8)
        proj_dict = get_projections(scaled_feature_dict, device)
        # proj_mats[alpha] = proj_dict

        # Apply projection to model weights
        model_projected = copy.deepcopy(model).to(device)
        model_projected.project_weights(proj_dict, project_classifier_head)
        # projected_models[alpha] = model_projected
        
        metrics_val, _, _ = evaluate_model(model_projected, clean_samples_dl, device)
        if metrics_val['ACC'] > best_ACC:
            best_alpha = alpha
            best_model = copy.deepcopy(model_projected)
            best_ACC = metrics_val['ACC']
        print("Val: ", metrics_val)
        
        metrics_test, _, _ = evaluate_model(model_projected, test_dl, device)
        print("Test:", metrics_test)
        
        results_dict[alpha] = OrderedDict()
        results_dict[alpha]['Val'] = metrics_val
        results_dict[alpha]['Test'] = metrics_test

    results_dict['Best Alpha'] = best_alpha
    print("\nSAP projection completed.")
    return results_dict, best_model



def SAP_unlearning_poison(
        model,
        clean_samples_dl,
        triggered_samples_dl,
        test_dl,
        device,
        project_classifier_head = True,
        scale_coff_list = [1000, 5000, 10000, 30000, 50000, 100000, 300000, 500000, 1000000],
        ):
    """
    Applies the SAP algorithm using a clean held-out dataset to compute activation projections.

    Args:
        model: trained (possibly noisy) model
        clean_samples_dl: DataLoader of clean / trusted samples
        device: torch device
        scale_coff_list: list of alpha values for SAP scaling
        prev_recur_proj_mat: optional previous round projectors (for recurrent SAP)
    Returns:
        projected_models: dict { alpha : model_copy_with_projection }
        proj_mats: dict { alpha : projector_matrices_per_layer }
    """

    model.to(device)
    model.eval()
    
    best_alpha = None
    best_model = None
    best_ASR = np.inf

    # Step 1: Collect activation representations on the clean dataset
    print("\nCollecting activation representations on clean set...")
    mat_retain_dict = get_representation_matrix(
        model, device, clean_samples_dl, set_name="Clean Set"
    )

    # Step 2: Compute SVDs (U, S) per layer
    print("\nComputing SVDs for each layer...")
    full_feature_retain_dict, full_s_retain_dict = get_SVD(
        mat_retain_dict, set_name="SVD Clean Set"
    )

    results_dict = OrderedDict()
    # Step 3: Loop over alphas to build projectors and update weights
    print("\nRunning SAP projection...")
    for alpha in scale_coff_list:
        print(f"  α = {alpha}")

        # Build scaled feature matrices (U Λ^{1/2})
        scaled_feature_dict = get_scaled_feature_mat(
            full_feature_retain_dict, full_s_retain_dict,
            alpha=alpha, device=device
        )

        # Compute projection matrices Mr = U Λ Uᵀ (Eq.8)
        proj_dict = get_projections(scaled_feature_dict, device)
        # proj_mats[alpha] = proj_dict

        # Apply projection to model weights
        model_projected = copy.deepcopy(model).to(device)
        model_projected.project_weights(proj_dict, project_classifier_head)
        # projected_models[alpha] = model_projected
        
        metrics_val, _, _ = evaluate_model(model_projected, triggered_samples_dl, device)
        if metrics_val['ACC'] < best_ASR:
            best_alpha = alpha
            best_model = copy.deepcopy(model_projected)
            best_ASR = metrics_val['ACC']
        print("Val: ", metrics_val)
        
        metrics_test, _, _ = evaluate_model(model_projected, test_dl, device)
        print("Test: ", metrics_test)
        
        results_dict[alpha] = OrderedDict()
        results_dict[alpha]['Val'] = metrics_val
        results_dict[alpha]['Test'] = metrics_test

    results_dict['Best Alpha'] = best_alpha
    print("\nSAP projection completed.")
    return results_dict, best_model





