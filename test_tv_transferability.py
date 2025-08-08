import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer, utils as trainer_utils
import matplotlib.pyplot as plt
import seaborn as sns

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




from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient
    
    
def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        base_alpha = 1.0 if tv_name == 'Gold TV' else -1.0
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
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg_t)
    
    base_model = model_factory.create_model(cfg_t['model'], num_classes)
    
    
    results_dir = results_dir / f"{cfg_s_name}_to_{cfg_t_name}"
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
        finetune_s_tvs[f"{float(ft_expr.split('_')[0])*100:.0f}% Noise, {ft_expr.split('_')[1]} Seed"] = TaskVector(pretrain_s_weights, ft_weight)
    finetune_s_tvs['Gold TV'] = ft_s_gold_tv
    finetune_s_tvs['Average TV'] = TaskVector.mean(finetune_s_tvs)
    finetune_s_tvs['Average TV Pruned 0.8'] = finetune_s_tvs['Average TV'].prune_small_weights(rate=0.8)
    
    ft_s_tvs_list = list(finetune_s_tvs.values())
    print(finetune_s_tvs.keys())
    tv_names = list(finetune_s_tvs.keys())
    
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
    results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names[-3:], ft_s_tvs_list[-3:])), results_dict, cfg_t, dataset, num_classes, gpu)
    
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