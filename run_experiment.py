import os
PYTHON_HASH_SEED = 0
os.environ["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import comet_ml
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
torch.set_float32_matmul_precision("high")

from src.datasets import dataset_factory, dataset_wrappers
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer, GradientAscentTrainer, utils as trainer_utils
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import misc_utils

import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, Subset, ConcatDataset
from functools import partial
from pathlib import Path
import pickle
import argparse
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
import math


from src.utils import embedding_space_analysis
from helper_funcs import evaluate_model, eval_model_on_clean_corrupted_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis

def initialize_model_dataset(experiment_type:str, architecture:str, cfg: dict):
    dataset_cfg = cfg['dataset']
    
    if architecture == 'clip':
        
        base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        base_model = model_factory.create_model(cfg['model'])
        base_model.freeze_all_heads()
        
        dataset_cfg['train_transforms'] = base_model.get_train_transforms()
        dataset_cfg['val_transforms'] = base_model.get_val_transforms()
        base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        
    elif architecture == 'dino':
        base_model = model_factory.create_model(cfg['model'])
        dataset_cfg['train_transforms'] = base_model.get_train_transforms()
        dataset_cfg['val_transforms'] = base_model.get_val_transforms()
        base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    elif architecture == 'regular':
        augmentations = None
        if cfg['dataset']['name'] == 'cifar10':
            augmentations = [
                transformsv2.RandomCrop(224, padding=4),
            ]
            # augmentations = [
            #     transformsv2.RandomCrop(32, padding=4),
            #     transformsv2.RandomHorizontalFlip(),
            # ]
        elif cfg['dataset']['name'] == 'cifar100':
            augmentations = [
                transformsv2.RandomCrop(224, padding=4),
            ]
            # augmentations = [
            #     transformsv2.RandomCrop(32, padding=4),
            #     transformsv2.RandomHorizontalFlip(),
            # ]
        elif cfg['dataset']['name'] == 'mnist':
            pass
        base_dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'], augmentations)
        base_model = model_factory.create_model(cfg['model'], num_classes)


    cfg['trainer']['mix']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['oracle']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['proxy']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['CF']['comet_api_key'] = os.getenv("COMET_API_KEY")

    return base_model, base_dataset, cfg



def inject_corruption(experiment_type:str, base_dataset, cfg: dict):
    strategy = cfg['strategy']
    if experiment_type == 'noise':
        base_dataset.inject_noise(**strategy['corruption']['mix'])
    elif experiment_type == 'IC':
        base_dataset.inject_noise(**strategy['corruption']['mix'])
    elif experiment_type == 'poison':
        base_dataset.inject_poison(**strategy['corruption']['mix'])
    
    return base_dataset
    
def finetune_models(experiment_type:str, architecture:str, outputs_dir: Path, cfg: dict, cfg_name:str):
    
    base_model, base_dataset, cfg = initialize_model_dataset(experiment_type, architecture, cfg)
    
    strategy = cfg['strategy']
    base_dataset = inject_corruption(experiment_type, base_dataset, cfg)
    

    
    if not outputs_dir.joinpath(f"{cfg_name}/mix/weights/weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
            
        experiment_name = f"{cfg_name}/mix"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['mix'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("weights.pth"))
        
      
    if not outputs_dir.joinpath(f"{cfg_name}/oracle/weights/weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/oracle"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['oracle'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("weights.pth"))
        
        
    # Catastrophic Forgetting baseline
    if not outputs_dir.joinpath(f"{cfg_name}/CF/weights/weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/weights.pth')
        checkpoint = torch.load(mix_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        experiment_name = f"{cfg_name}/CF"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        # For asymmetric label noise and IC we only finetune on samples that actually are corrupted with corruption kernel
        # using their clean labels.
        if experiment_type == 'poison':
            proxy_conf = copy.deepcopy(strategy['corruption']['proxy'][0])
            proxy_conf['set'] = 'Heldout'
            proxy_conf['rate'] = 0.0
            dataset.inject_poison(**proxy_conf)
            dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
        elif experiment_type == 'noise':
            proxy_conf = strategy['corruption']['proxy'][0]
            
            if proxy_conf['noise_type'] == 'symmetric':
                dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
            elif proxy_conf['noise_type'] == 'asymmetric':
                proxy_conf['set'] = 'Heldout'
                dataset.inject_noise(**proxy_conf)
                hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
                dataset.switch_labels_to_clean(hs_noisy)
                dataset.set_trainset(hs_noisy, shuffle=True)
                 
        elif experiment_type == 'IC':
            proxy_conf = strategy['corruption']['proxy'][0]
            proxy_conf['set'] = 'Heldout'
            dataset.inject_noise(**proxy_conf)
            hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
            dataset.switch_labels_to_clean(hs_noisy)
            dataset.set_trainset(hs_noisy, shuffle=True)
        
        
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['CF'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("weights.pth"))
        

        
    
    for idx, proxy_conf in enumerate(strategy['corruption']['proxy']):
        if not outputs_dir.joinpath(f"{cfg_name}/proxy_{proxy_conf['seed']}/weights/weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            experiment_name = f"{cfg_name}/proxy_{proxy_conf['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            
            if experiment_type == 'poison': 
                # Exclude clean samples from target class
                proxy_conf['set'] = 'Heldout'
                dataset.inject_poison(**proxy_conf)
                clean_ho_ds, poinsoned_ho_ds = dataset.get_clean_noisy_subsets('Heldout')
                dataset.set_trainset(poinsoned_ho_ds, shuffle=True)
                
            elif experiment_type == 'noise':
                if proxy_conf['noise_type'] == 'symmetric':
                    dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
                    dataset.inject_noise(**proxy_conf)
                # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
                elif proxy_conf['noise_type'] == 'asymmetric':
                    proxy_conf['set'] = 'Heldout'
                    dataset.inject_noise(**proxy_conf)
                    hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
                    dataset.set_trainset(hs_noisy, shuffle=True)
                    
            elif experiment_type == 'IC':
                # For IC, we only consider the noisy samples (only the pair of classes swapped.)
                proxy_conf['set'] = 'Heldout'
                dataset.inject_noise(**proxy_conf)
                hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
                dataset.set_trainset(hs_noisy, shuffle=True)
                

                
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['proxy'],
                exp_name=experiment_name,
                exp_tags=None,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            torch.save(model.state_dict(), weights_dir / Path("weights.pth"))  
            



def apply_tv(experiment_type:str, architecture:str, outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    results_dirs['cms'] = results_dir / 'confusion_mats'
    results_dirs['Ts'] = results_dir / 'transition_mats'
    results_dirs['W_norms'] = results_dir / 'weight_norms'
    results_dirs['TV_norms'] = results_dir / 'TV_norms'
    results_dirs['embed_plots'] = results_dir / 'embedding_plots'
    results_dirs['metrics'] = results_dir / 'metrics'
    for dir in results_dirs.values():
        dir.mkdir(exist_ok=True, parents=True)
    
    
    model, base_dataset, cfg = initialize_model_dataset(experiment_type, architecture, cfg)
    base_dataset.reset_train_dl(shuffle=False)
    
    strategy = cfg['strategy']
    

    
    
    if experiment_type == 'poison':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_poison(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        dataset_corrupted.inject_poison(**proxy_conf)
        # Exclude clean samples from target class
        clean_ho_ds, poinsoned_ho_ds = dataset_corrupted.get_clean_noisy_subsets('Heldout')
        dataset_corrupted.set_heldoutset(poinsoned_ho_ds)
        
    elif experiment_type == 'noise':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_noise(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        if proxy_conf['noise_type'] == 'symmetric':
            dataset_corrupted.inject_noise(**proxy_conf)
        elif proxy_conf['noise_type'] == 'asymmetric':
            dataset_corrupted.inject_noise(**proxy_conf)
            hs_clean, hs_noisy = dataset_corrupted.get_clean_noisy_subsets(set='Heldout')
            dataset_corrupted.switch_labels_to_clean(hs_noisy)
            dataset_clean.set_heldoutset(copy.deepcopy(hs_noisy), shuffle=False)
            dataset_corrupted.switch_labels_to_noisy(hs_noisy)
            dataset_corrupted.set_heldoutset(hs_noisy, shuffle=False)
                
    elif experiment_type == 'IC':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_noise(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        
        dataset_corrupted.inject_noise(**proxy_conf)
        hs_clean, hs_noisy = dataset_corrupted.get_clean_noisy_subsets(set='Heldout')
        dataset_corrupted.switch_labels_to_clean(hs_noisy)
        dataset_clean.set_heldoutset(copy.deepcopy(hs_noisy), shuffle=False)
        dataset_corrupted.switch_labels_to_noisy(hs_noisy)
        dataset_corrupted.set_heldoutset(hs_noisy, shuffle=False)
        
        

    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
        

    # Load weights while removing classifier head's weights from the state dict for CLIP
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    oracle_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"oracle/weights/weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    CF_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"CF/weights/weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    
    proxy_weights = OrderedDict()
    
    for proxy_conf in strategy['corruption']['proxy']:
        ft_expr_dir = outputs_dir / f"proxy_{proxy_conf['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        proxy_weights[f"Proxy Seed {proxy_conf['seed']}"] = n_weights
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in proxy_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Proxy'] = only_tv
    else:
        task_vectors['Proxy'] = TaskVector.mean(task_vectors)
        
    
    task_vectors['CF'] = TaskVector(mix_weights, CF_weights)
    
    
    task_vectors['Random Vector'] = task_vectors['Proxy'].generate_random_vector_with_same_layer_norms(seed=20)
    task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    task_vectors['Oracle'] = TaskVector(pt_weights, oracle_weights)


    TV_norms = OrderedDict()
    for name, tv in task_vectors.items():
        TV_norms[name] = tv.norm().item()
    with open(results_dirs['TV_norms'] / 'norms.json' , 'w') as json_file:
        json.dump(TV_norms, json_file, indent=4)
    
    ft_tvs_list = list(task_vectors.values())
    tv_names = list(task_vectors.keys())

    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
    with open(results_dirs['cms'] / "tv_sim.pkl", "wb") as f:
        pickle.dump(task_sim, f)
    
    misc_utils.plot_confusion_matrix(
        title='Task Vector Similarity Matrix',
        cm=task_sim,
        class_names=tv_names,
        color_map='vlag',
        color_bar=True,
        vmin= -1.0,
        vmax= 1.0,
        x_label='Task Vectors',
        y_label='Task Vectors',
        tick_label_font_size=6,
        filepath=results_dir / 'task_similarities.png',
        show=False
    )



    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():

        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset_clean.get_test_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset_corrupted, gpu)
        
        
        model.load_state_dict(oracle_weights, strict=False)
        oracle_test_results, _, _ = evaluate_model(model, dataset_clean.get_test_dataloader(), gpu)
        oracle_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset_corrupted, gpu)
        
        model.load_state_dict(CF_weights, strict=False)
        CF_test_results, _, _ = evaluate_model(model, dataset_clean.get_test_dataloader(), gpu)
        CF_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset_corrupted, gpu)
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
        results_dict['Oracle'] = {'test_results': oracle_test_results, 'train_results': oracle_train_results}
        results_dict['CF'] = {'test_results': CF_test_results, 'train_results': CF_train_results}
        
        
        if experiment_type == 'poison':
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        elif experiment_type == 'noise':
            alphas = tqdm(np.round(np.linspace(-0.05, -3.0, 60), 2))
        elif experiment_type == 'IC':
            alphas = tqdm(np.round(np.linspace(-0.05, -1.5, 30), 2))
        for alpha in alphas:
            
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Proxy'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset_clean.get_test_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset_corrupted, gpu)
            
            if experiment_type == 'poison':
                tv_ho_resutls, _, _ = evaluate_model(model, dataset_corrupted.get_heldout_dataloader(), gpu)
                results_dict[alpha] = {'test_results': tv_test_results, 'ho_results': tv_ho_resutls, 'train_results': tv_train_results}
            elif experiment_type == 'IC':
                tv_ho_resutls, _, _ = evaluate_model(model, dataset_clean.get_heldout_dataloader(), gpu)
                results_dict[alpha] = {'test_results': tv_test_results, 'ho_results': tv_ho_resutls, 'train_results': tv_train_results}
            else:
                results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    
            
    if 'alpha_hat' not in results_dict: 
        if experiment_type == 'noise':  
            proxy_conf = strategy['corruption']['proxy'][0]
            if dataset_clean.dataset_name == 'MNIST':
                if proxy_conf['noise_type'] == 'asymmetric':
                    coverage_rate = 0.5
                else:
                    coverage_rate = 1.0
            elif dataset_clean.dataset_name == 'CIFAR10':
                if proxy_conf['noise_type'] == 'asymmetric':
                    coverage_rate = 0.5
                else:
                    coverage_rate = 1.0
            elif dataset_clean.dataset_name == 'CIFAR100':
                coverage_rate = 0.95
            
            num_clusters = dataset_clean.get_num_classes()
            alpha_est_support_dl = dataset_clean.get_heldout_dataloader()
            alpha_est_support_size = len(dataset_clean.get_heldoutset())
            ideal_cluster_balance = alpha_est_support_size / num_clusters
            num_neighbor_agr_check = math.floor(ideal_cluster_balance / 2)
            

            from estimate_alpha import select_alpha_by_knn_self_agreement
            alpha_hat = select_alpha_by_knn_self_agreement(
                model=model,
                feature_extractor=model.get_feature_extractor(),
                classifier=model.get_active_head(),
                state0=mix_weights,
                taskvector=task_vectors['Proxy'],
                unlabeled_loader=alpha_est_support_dl,
                num_clusters=num_clusters,
                k=num_neighbor_agr_check,
                coverage_rate=coverage_rate,
                alphas=np.round(np.linspace(-0.0, -2.0, 41), 2),
                device=gpu
            )
            results_dict['alpha_hat'] = alpha_hat
            
        elif experiment_type == 'poison':
            forget_rate_thrsh = {
                'MNIST': 0.01,
                'CIFAR10': 0.01,
                'CIFAR100': 0.01
            }
            alphas = np.round(np.linspace(-0.05, -2.0, 40), 2)
            alpha_hat = 0.0
            for alpha in alphas:
                metrics = results_dict.get(alpha, None)
                if not metrics: metrics = results_dict.get(str(alpha), None)
                if not metrics: print('alpha not found', alpha)
                if round(metrics['ho_results']['ACC'], 2) <= forget_rate_thrsh[dataset_clean.dataset_name]:
                    alpha_hat = alpha
                    break
            results_dict['alpha_hat'] = alpha_hat
        
        
        elif experiment_type == 'IC':
            alphas = np.round(np.linspace(-0.05, -1.5, 30), 2)
            base_IC_acc = results_dict['Mix']['ho_results']['ACC']
            alpha_hat = 0.0
            for alpha in alphas:
                metrics = results_dict.get(alpha, None)
                if not metrics: metrics = results_dict.get(str(alpha), None)
                if not metrics: print('alpha not found', alpha)
                if metrics['ho_results']['ACC'] >= base_IC_acc:
                    alpha_hat = alpha
                    base_IC_acc = metrics['ho_results']['ACC'] 
            
            results_dict['alpha_hat'] = alpha_hat
        
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
        

    if 'Random Vector' not in results_dict:
        model.load_state_dict(mix_weights, strict=False)
        alpha_hat = results_dict['alpha_hat']
        task_vectors['Random Vector'].apply_to(model, scaling_coef=alpha_hat, strict=False)
        random_test_results, _, _ = evaluate_model(model, dataset_clean.get_test_dataloader(), gpu)
        random_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset_corrupted, gpu)
        
        if experiment_type == 'poison':
            random_ho_resutls, _, _ = evaluate_model(model, dataset_corrupted.get_heldout_dataloader(), gpu)
            results_dict['Random Vector'] = {'test_results': random_test_results, 'ho_results': random_ho_resutls, 'train_results': random_train_results}
        elif experiment_type == 'IC':
            random_ho_resutls, _, _ = evaluate_model(model, dataset_clean.get_heldout_dataloader(), gpu)
            results_dict['Random Vector'] = {'test_results': random_test_results, 'ho_results': random_ho_resutls, 'train_results': random_train_results}
        else:
            results_dict['Random Vector'] = {'test_results': random_test_results, 'train_results': random_train_results}
        
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
        
        

    
    # figs_alpha, fig_gold = embedding_space_analysis.pca_evolution_plot(
    #     model=model,
    #     base_weights=mix_weights,
    #     gold_weights=None,
    #     dataset=dataset_clean,
    #     task_vector=task_vectors['Average'],
    #     split='Test',
    #     # alpha_range=np.round(np.linspace(0.0, results_dict['alpha_KNN'], 4) / 0.05) * 0.05,
    #     alpha_range=np.linspace(0.0, results_dict['alpha_KNN'], 60),
    #     device=gpu,
    #     saving_dir=results_dirs['embed_plots']
    # )
    



def apply_SAP(experiment_type:str, architecture:str, outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    results_dirs['cms'] = results_dir / 'confusion_mats'
    results_dirs['Ts'] = results_dir / 'transition_mats'
    results_dirs['W_norms'] = results_dir / 'weight_norms'
    results_dirs['TV_norms'] = results_dir / 'TV_norms'
    results_dirs['embed_plots'] = results_dir / 'embedding_plots'
    results_dirs['metrics'] = results_dir / 'metrics'
    for dir in results_dirs.values():
        dir.mkdir(exist_ok=True, parents=True)
    
    
    base_model, base_dataset, cfg = initialize_model_dataset(experiment_type, architecture, cfg)
    base_dataset.reset_train_dl(shuffle=False)
    
    strategy = cfg['strategy']
    
    if experiment_type == 'poison':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_poison(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        dataset_corrupted.inject_poison(**proxy_conf)
        # Exclude clean samples from target class
        clean_ho_ds, poinsoned_ho_ds = dataset_corrupted.get_clean_noisy_subsets('Heldout')
        dataset_corrupted.set_heldoutset(poinsoned_ho_ds)
        
    elif experiment_type == 'noise':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_noise(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        if proxy_conf['noise_type'] == 'symmetric':
            dataset_corrupted.inject_noise(**proxy_conf)
        elif proxy_conf['noise_type'] == 'asymmetric':
            dataset_corrupted.inject_noise(**proxy_conf)
            hs_clean, hs_noisy = dataset_corrupted.get_clean_noisy_subsets(set='Heldout')
            dataset_corrupted.switch_labels_to_clean(hs_noisy)
            dataset_clean.set_heldoutset(copy.deepcopy(hs_noisy), shuffle=False)
            dataset_corrupted.switch_labels_to_noisy(hs_noisy)
            dataset_corrupted.set_heldoutset(hs_noisy, shuffle=False)
                
    elif experiment_type == 'IC':
        dataset_clean = copy.deepcopy(base_dataset)
        dataset_corrupted = copy.deepcopy(base_dataset)
        dataset_corrupted.inject_noise(**strategy['corruption']['mix'])
        
        proxy_conf = strategy['corruption']['proxy'][0]
        proxy_conf['set'] = 'Heldout'
        
        dataset_corrupted.inject_noise(**proxy_conf)
        hs_clean, hs_noisy = dataset_corrupted.get_clean_noisy_subsets(set='Heldout')
        dataset_corrupted.switch_labels_to_clean(hs_noisy)
        dataset_clean.set_heldoutset(copy.deepcopy(hs_noisy), shuffle=False)
        dataset_corrupted.switch_labels_to_noisy(hs_noisy)
        dataset_corrupted.set_heldoutset(hs_noisy, shuffle=False)
    
    
    
    
    
    # Load weights while removing classifier head's weights from the state dict for CLIP
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    
    base_model.load_state_dict(mix_weights)
    
    sap_model = copy.deepcopy(base_model)
    
    
    from src.trainers import sap_unlearner
    
    
    
    if experiment_type == 'noise':
        sap_unlearner.SAP_unlearning_noise(
            model=sap_model,
            clean_samples_dl=dataset_clean.get_heldout_dataloader(),
            project_classifier_head=True,
            device=gpu
        )
    elif experiment_type == 'poison':
        sap_unlearner.SAP_unlearning_poison(
            model=sap_model,
            clean_samples_dl=dataset_clean.get_heldout_dataloader(),
            triggered_samples_dl=dataset_corrupted.get_heldout_dataloader(),
            project_classifier_head=True,
            device=gpu
        )
    

    
    

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-e",
        "--experiment",
        help="Which experiment to run.",
        type=str,
        choices=['noise', 'IC', 'poison']
    )
    
    parser.add_argument(
        "-a",
        "--arch",
        help="Model architecture.",
        type=str,
        choices=['clip', 'dino', 'regular']
    )
    
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the image encoder with forzen heads on noisy datasets.",
        action="store_true",
    )
    
    
    parser.add_argument(
        "-t",
        "--tv",
        help="Apply task vectors to an already trained and finetuned model.",
        action="store_true",
    )
    
    
    parser.add_argument(
        "-s",
        "--sap",
        help="Apply SAP unlearning.",
        action="store_true",
    )
    
    
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    
    expr_arch = Path(f"single_experiment/{args.arch}_{args.experiment}_TA")
    
    
    cfg_path = Path("configs").absolute() / expr_arch / f"{args.config}.yaml"
    
    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/").absolute() / expr_arch
    results_dir = Path("results/").absolute() / expr_arch
    results_dir.mkdir(exist_ok=True, parents=True)
    
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])

        
    if args.finetune:
        finetune_models(args.experiment, args.arch, outputs_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv:
        apply_tv(args.experiment, args.arch, outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    
    if args.sap:
        apply_SAP(args.experiment, args.arch, outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)


if __name__ == "__main__":
    main()