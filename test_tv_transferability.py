import comet_ml
from src.datasets import dataset_factory, data_utils
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
import torch
import torchmetrics
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
from collections import OrderedDict




def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch



def evaluate_model(model, dataloader, device):
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


def search_optimal_coefficient(base_model, task_vector, search_range, dataset, num_classes, device):
    """
    Performs a search to find the optimal task vector scaling coefficient.

    Args:
        base_model (torch.nn.Module): The pre-trained model. A deepcopy is made for each evaluation.
        task_vector (TaskVector): The task vector object.
        dataset: The dataset object to get the test dataloader from.
        search_range (list or tuple): A list/tuple [min_val, max_val] for the search.
        device (torch.device): The device to run evaluation on.
        num_classes (int): The number of classes for the confusion matrix.

    Returns:
        tuple: (best_coefficient, best_performance_metrics, confusion_matrix_tensor)
    """
    test_dataloader = dataset.get_test_dataloader()
    
    best_coef = 0.0
    best_acc = -1.0
    best_results = {}
    
    print("--- Starting Coarse Search ---")
    coarse_search_grid = np.arange(search_range[0], search_range[1] + 0.1, 0.1)
    
    for scale_coef in tqdm(coarse_search_grid, desc="Coarse Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
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
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results

    # print(f"\nRecalculating metrics and confusion matrix for best coefficient: {best_coef:.2f}")
    final_model = copy.deepcopy(base_model)
    task_vector.apply_to(final_model, scaling_coef=best_coef)
    final_model.to(device)

    best_results, all_preds, all_targets = evaluate_model(final_model, test_dataloader, device)
    
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    best_cm_tensor = confmat_metric(all_preds, all_targets)

    return best_coef, best_results, best_cm_tensor


def eval_model_on_clean_noise_splits(model, cfg, dataset, device):
    dataset_cpy = copy.deepcopy(dataset)
    strategy = cfg['strategy']
    dataset_cpy.inject_noise(**strategy['noise']['pretraining'])
    clean_set, noisy_set = dataset_cpy.get_clean_noisy_subsets(set='Train')
    
    dataset_cpy.set_trainset(clean_set, shuffle=False)
    clean_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    noisy_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dummy_instance = noisy_set
    while not isinstance(dummy_instance, data_utils.NoisyClassificationDataset):
        dummy_instance = dummy_instance.dataset
    dummy_instance.switch_to_clean_lables()
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    healing_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
        'noisy_set': noisy_metric,
        'healing_noise': healing_metric,
    }
    
    
    
    
def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        base_alpha = 1.0 if tv_name == 'Gold' else -1.0
        results[tv_name][base_alpha] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=base_alpha)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
        base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][base_alpha]['test_results'] = base_test_results
        results[tv_name][base_alpha]['train_results'] = base_train_split_results
        
        base_model = copy.deepcopy(model)

        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=tv,
            search_range=(0.0, 3.0) if tv_name == 'Gold' else (-3.0, 0.0),
            dataset=dataset,
            num_classes=num_classes,
            device=device
        )
        
        results[tv_name][best_coef] = OrderedDict()
        results[tv_name][best_coef]['test_results'] = best_results
        
        tv.apply_to(base_model, scaling_coef=best_coef)
        
        after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][best_coef]['train_results'] = after_tv_metrics
        
    
    return results


            
def transfer_noise_vectors(outputs_dir: Path, results_dir: Path, cfg_s: dict, cfg_s_name:str, cfg_t: dict, cfg_t_name:str):
    training_seed = cfg_t['training_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg_t)
    
    base_model = model_factory.create_model(cfg_t['model'], num_classes)
    
    
    results_dir = results_dir / cfg_t_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    base_target_expr_dir = outputs_dir / cfg_t_name
    pretrain_t_dir = base_target_expr_dir / 'pretrain'
    pretrain_t_weights = torch.load(pretrain_t_dir / 'weights/model_weights.pth', map_location=cpu)
    
    
    
    base_source_expr_dir = outputs_dir / cfg_s_name
    pretrain_s_dir = base_source_expr_dir / 'pretrain'
    ft_gold_s_dir = base_source_expr_dir / 'finetune_gold'
    finetune_s_dirs = OrderedDict()
    for idx, noise_tv in enumerate(cfg_s['strategy']['noise']['finetuning']):
        ft_expr_dir = base_source_expr_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        finetune_s_dirs[f"{noise_tv['noise_rate']}_{noise_tv['seed']}"] = ft_expr_dir
        
    pretrain_s_weights = torch.load(pretrain_s_dir / 'weights/model_weights.pth', map_location=cpu)
    ft_gold_s_wieghts = torch.load(ft_gold_s_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_s_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_s_dirs.items():
        finetune_s_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
    
    
    
    ft_s_gold_tv = TaskVector(pretrain_s_weights, ft_gold_s_wieghts)
    finetune_s_tvs = OrderedDict()
    for ft_expr, ft_weight in finetune_s_weights.items():
        finetune_s_tvs[ft_expr] = TaskVector(pretrain_s_weights, ft_weight)
    finetune_s_tvs['avg_noise'] = TaskVector.mean(finetune_s_tvs)

    ft_s_tvs_list = [ft_s_gold_tv]
    ft_s_tvs_list.extend(list(finetune_s_tvs.values()))
    print(finetune_s_tvs.keys())
    
    tv_names = ['Gold']
    tv_names.extend([f"{float(n_s.split('_')[0])*100:.0f}% Noise, {n_s.split('_')[1]} Seed" for n_s in list(finetune_s_tvs.keys())[:-1]])
    tv_names.extend(['Average TV'])
    
    # task_sim = []
    # for i in range(len(ft_s_tvs_list)):
    #     anchor_tv = ft_s_tvs_list[i]
    #     task_sim.append([])
    #     for j in range(len(ft_s_tvs_list)):
    #         other_tv = ft_s_tvs_list[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         task_sim[i].append(cos_sim)
    # task_sim = np.array(task_sim)
    
    # misc_utils.plot_confusion_matrix(
    #     title='Task Vector Similarity Matrix',
    #     cm=task_sim,
    #     class_names=tv_names,
    #     color_map='vlag',
    #     color_bar=True,
    #     vmin= -1.0,
    #     vmax= 1.0,
    #     x_label='Task Vectors',
    #     y_label='Task Vectors',
    #     tick_label_font_size=6,
    #     filepath=results_dir / 'task_similarities.png',
    #     show=False
    # )
    
    
    base_model.load_state_dict(pretrain_t_weights)
    results_dict = OrderedDict()    
    results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names, ft_s_tvs_list)), results_dict, cfg_t, dataset, num_classes, gpu)
    
    print(results_dict)
    
    
    with open(results_dir / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    # generate_latex_table_from_results(results_dict, results_dir / 'results_tex.txt')
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Config to take the noise vectors from.",
        type=str,
    )
    
    parser.add_argument(
        "-t",
        "--transfer",
        help="Config to which the nosie vectors are applied.",
        type=str
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_s_path = Path('configs/single_experiment/pretrain_on_noisy') / f"{args.config}.yaml"
    if not cfg_s_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_s_path, 'r') as file:
        cfg_s = yaml.full_load(file)
        
    cfg_t_path = Path('configs/single_experiment/pretrain_on_noisy') / f"{args.transfer}.yaml"
    if not cfg_t_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_t_path, 'r') as file:
        cfg_t = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/pretrain_on_noisy").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/noise_tv_transferability").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    
    transfer_noise_vectors(
        outputs_dir,
        results_dir,
        cfg_s=cfg_s,
        cfg_s_name=cfg_s_path.stem,
        cfg_t=cfg_t,
        cfg_t_name=cfg_t_path.stem
    )