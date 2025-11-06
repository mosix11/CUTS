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
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
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
        noise_weights[f"Seed {noise_tv['seed']}"] = n_weights
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average'] = only_tv
    else:
        task_vectors['Average'] = TaskVector.mean(task_vectors)
        
    
    task_vectors['CF'] = TaskVector(mix_weights, ft_ho_clean_weights)
    
    # task_vectors['Gold'] = TaskVector(pt_weights, gold_weights)
    task_vectors['Random Vector'] = task_vectors['Average'].generate_random_vector_with_same_layer_norms(seed=20)
    task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    
    # with open(results_dir / "metrics.json", "r") as json_file:
    #     results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
    # estimated_noise_vector =  task_vectors['Average'] * (-1 * results_dict['alpha_KNN'])
    # estimated_clean_vector = task_vectors['Mix'] - estimated_noise_vector
    # task_vectors['Clean'] = estimated_clean_vector


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



    # results_mtl_dict = OrderedDict()
    # if not results_dir.joinpath('metrics_mtl.json').exists():
    #     alphas = tqdm(np.round(np.linspace(0.0, 1.0, 21), 2))
        
    #     for alpha in alphas:
    #         model.load_state_dict(pt_weights, strict=False)
    #         task_vectors['Mix'].apply_to(model, scaling_coef=alpha, strict=False)
    #         tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    #         tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

    #         results_mtl_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    #     with open(results_dir / 'metrics_mtl.json' , 'w') as json_file:
    #         json.dump(results_mtl_dict, json_file, indent=4)
    

    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():

        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        
        model.load_state_dict(gold_weights, strict=False)
        gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        gold_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        model.load_state_dict(ft_ho_clean_weights, strict=False)
        ft_ho_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        ft_ho_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
        results_dict['Gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
        results_dict['FT HO Clean'] = {'test_results': ft_ho_test_results, 'train_results': ft_ho_train_results}
        
        if strategy['noise']['finetuning'][0]['noise_type'] == 'asymmetric':
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        else:
            alphas = tqdm(np.round(np.linspace(-0.05, -3.0, 60), 2))
        for alpha in alphas:
            
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

            results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    if 'alpha_KNN' not in results_dict:        
        num_clusters = dataset_clean.get_num_classes()
        alpha_est_support_dl = dataset_clean.get_heldout_dataloader()
        alpha_est_support_size = len(dataset_clean.get_heldoutset())
        ideal_cluster_balance = alpha_est_support_size / num_clusters
        num_neighbor_agr_check = math.floor(ideal_cluster_balance / 2)
        if dataset.dataset_name == 'MNIST':
            coverage_rate = 1.0
        elif dataset.dataset_name == 'CIFAR10':
            coverage_rate = 1.0
        elif dataset.dataset_name == 'CIFAR100':
            coverage_rate = 0.95

        from estimate_alpha import select_alpha_by_knn_self_agreement
        alpha_kNN = select_alpha_by_knn_self_agreement(
            model=model,
            feature_extractor=model.get_feature_extractor(),
            classifier=model.get_active_head(),
            state0=mix_weights,
            taskvector=task_vectors['Average'],
            unlabeled_loader=alpha_est_support_dl,
            num_clusters=num_clusters,
            k=num_neighbor_agr_check,
            coverage_rate=coverage_rate,
            alphas=np.round(np.linspace(-0.0, -2.0, 41), 2),
            device=gpu
        )
        
        results_dict['alpha_KNN'] = alpha_kNN
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    

    if 'Random Vector' not in results_dict:
        model.load_state_dict(mix_weights, strict=False)
        alpha_kNN = results_dict['alpha_KNN']
        task_vectors['Random Vector'].apply_to(model, scaling_coef=alpha_kNN, strict=False)
        random_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        random_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        results_dict['Random Vector'] = {'test_results': random_test_results, 'train_results': random_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
        
        

    
    figs_alpha, fig_gold = embedding_space_analysis.pca_evolution_plot(
        model=model,
        base_weights=mix_weights,
        gold_weights=None,
        dataset=dataset_clean,
        task_vector=task_vectors['Average'],
        split='Test',
        # alpha_range=np.round(np.linspace(0.0, results_dict['alpha_KNN'], 4) / 0.05) * 0.05,
        alpha_range=np.linspace(0.0, results_dict['alpha_KNN'], 60),
        device=gpu,
        saving_dir=results_dirs['embed_plots']
    )
    
    exit()

    # Weight Space Disentanglemet Analysis
    # clean_train_ds, noisy_train_ds = dataset.get_clean_noisy_subsets('Train')
    # subset_size  = 2048
    # def random_subset(ds, k, seed: int):
    #     k = min(k, len(ds))
    #     g = torch.Generator().manual_seed(seed)
    #     idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    #     return Subset(ds, idx)

    # clean_subset = random_subset(clean_train_ds, subset_size, dataset_seed)
    # noisy_subset = random_subset(noisy_train_ds, subset_size, dataset_seed + 1)
    
    # with open(results_dir / "metrics_seed.json", "r") as json_file:
    #     results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
    
    # records = []
    # for a_str, res in results_dict['Seed 10'].items():
    #     if a_str in ['Mix', 'Gold', 'FT HO Clean']: continue
    #     a = float(a_str) if not isinstance(a_str, (int, float)) else a_str
    #     test_acc  = res["test_results"]["ACC"]
    #     test_loss = res["test_results"]["Loss"]
    #     noisy_acc = res["train_results"]["noisy_set"]["ACC"]
    #     records.append((a, test_acc, test_loss, noisy_acc))
    # alpha_max_test_acc = max(records, key=lambda x: x[1])[0]
    # alpha_min_test_loss = min(records, key=lambda x: x[2])[0]

    # forgetting_threshold = 0.09
    # alpha_forgetting_thrsh = None
    # for a, _, _, noisy_acc in sorted(records, key=lambda x: x[0], reverse=True):
    #     if noisy_acc <= forgetting_threshold:
    #         alpha_forgetting_thrsh = a
    #         break
        
    # print(
    #     'Alpha Max Test ACC:', alpha_max_test_acc,
    #     'Apha Min Test Loss:', alpha_min_test_loss,
    #     'Alpha Forget Threshold:', alpha_forgetting_thrsh
    #     )

    # estimated_noise_vector = task_vectors['Seed 10'] * alpha_forgetting_thrsh * -1 # alpha is negative
    # estimated_clean_vector = task_vectors['Mix'] - estimated_noise_vector
    estimated_noise_vector =  task_vectors['Average'] * (-1 * results_dict['alpha_KNN'])
    estimated_clean_vector = task_vectors['Mix'] - estimated_noise_vector

    
    # exit()
    
    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_analysis(
    #     model=model,
    #     taskvector1=clean_vector,
    #     support_tv1=clean_subset,
    #     taskvector2=noise_vector,
    #     support_tv2=noisy_subset,
    #     alhpa_range=(-3.0, 3.0),
    #     step=0.3,
    #     batch_size=512,
    #     device=gpu
    # )
    # with open(results_dir / "WD.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)
    
    subset_size  = 4096
    def random_subset(ds, k, seed: int):
        k = min(k, len(ds))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:k].tolist()
        return Subset(ds, idx)

    test_subset = random_subset(dataset.get_testset(), subset_size, dataset_seed)
    

    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_antitask_analysis(
    #     model=model,
    #     clean_tv=clean_vector,
    #     noise_tv=noise_vector,
    #     testset=test_subset,
    #     alpha_range=(0, 2),
    #     step=0.1,
    #     batch_size=512,
    #     device=gpu,
    #     metric='loss',
    # )
    # with open(results_dir / "WD_AT2_acc.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)
    
    
    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_antitask_analysis_acc(
        # model=model,
        # taskvector1=estimated_clean_vector,
        # taskvector2=estimated_noise_vector,
        # shared_support=test_subset,
        # calibration_dl=None,
        # alpha_range=(0.0, 2.5),
        # step=0.1,
        # batch_size=512,
        # device=gpu,
    # )
    # with open(results_dir / "WD2.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)
    
    from alignemnt_score import compute_task_vector_alignment
    
    alingment_score = compute_task_vector_alignment(
        model=model,
        clean_tv=estimated_clean_vector,
        corruption_tv=estimated_noise_vector,
        testset_tv1=test_subset,
        testset_tv2=None,
        dataset_name=dataset.dataset_name,
        corruption_type='sym',
        batch_size=512,
        device=gpu,
    )
    print(alingment_score)
    


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

if __name__ == "__main__":
    main()