import os
PYTHON_HASH_SEED = 0
os.environ["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import comet_ml
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
torch.set_float32_matmul_precision("high")

from src.datasets import dataset_factory, dataset_wrappers, BaseClassificationDataset
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
from helper_funcs import evaluate_model, prepare_batch
from src.utils import weight_norm_analysis
from WD_analysis import apply_WD_analysis, apply_WD_antitask_analysis, apply_WD_antitask_analysis_acc


def get_clean_corr_subsets(dataset):
    clean_indices = []
    noisy_indices = []
    poisoned_indices = []
    both_indices = []
    for item in dataset:
        x, y, idx, is_corr = item
        if is_corr == 1:
            noisy_indices.append(idx)
        elif is_corr == 2:
            poisoned_indices.append(idx)
        elif is_corr == 3:
            noisy_indices.append(idx)
            poisoned_indices.append(idx)
            both_indices.append(idx)
        elif is_corr == 0:
            clean_indices.append(idx)
        else:
            raise RuntimeError(f'Is corr value is {is_corr}')
    
    return Subset(dataset, clean_indices), Subset(dataset, noisy_indices), Subset(dataset, poisoned_indices), Subset(dataset, both_indices)

def eval_model_on_clean_corr_splits(
    model:torch.nn.Module,
    dataset:BaseClassificationDataset,
    device:torch.device
):
    dataset_cpy = copy.deepcopy(dataset)
    
    ds = dataset_cpy.get_trainset()

    clean_set, noisy_set, poisoned_set, both_set = get_clean_corr_subsets(ds)
    
    dataset_cpy.set_trainset(clean_set, shuffle=False)
    clean_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    noisy_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(poisoned_set, shuffle=False)
    poisoned_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(both_set, shuffle=False)
    both_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    
    # dummy_instance = noisy_set
    # while not isinstance(dummy_instance, (dataset_wrappers.NoisyClassificationDataset, dataset_wrappers.PoisonedClassificationDataset)):
    #     dummy_instance = dummy_instance.dataset
    # dummy_instance.switch_to_clean_lables()
    
    # dataset_cpy.set_trainset(noisy_set, shuffle=False)
    # healing_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
        'noisy_set': noisy_metric,
        # 'healing_noise': healing_metric,
        'poisoned_set': poisoned_metric,
        'both_set': both_metric,
    }



def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    dataset_cfg = cfg['datasets'][0]
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    strategy = cfg['strategy']
    # Inject noise first
    base_dataset.inject_noise(**strategy['noise']['pretraining'])
    # Inject poison next
    base_dataset.inject_poison(**strategy['poison']['pretraining'])
    
    # trainset = base_dataset.get_trainset()
    # clean_ds, corr_ds = base_dataset.get_clean_noisy_subsets('Train')
    
    # num_noisy = 0
    # num_poisoned = 0
    # num_both = 0
    # for item in corr_ds:
    #     x, y, idx, is_corr = item
    #     if is_corr == 1:
    #         num_noisy += 1
    #     elif is_corr == 2:
    #         num_poisoned += 1
    #     elif is_corr == 3:
    #         num_both += 1
            
    # print(num_noisy, num_poisoned, num_both)
    # exit()
    
    if not outputs_dir.joinpath(f"{cfg_name}/mix/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
            
        experiment_name = f"{cfg_name}/mix"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        finetuning_cfg = None
        if 'mix' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['mix']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
      
    if not outputs_dir.joinpath(f"{cfg_name}/clean/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/clean"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        finetuning_cfg = None
        if 'clean' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['clean']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
        

        

            
    for idx, poison_tv in enumerate(strategy['poison']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_p_{poison_tv['rate']}_{poison_tv['seed']}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
                    
            experiment_name = f"{cfg_name}/finetune_p_{poison_tv['rate']}_{poison_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            # Exclude clean samples from target class
            poison_tv['set'] = 'Heldout'
            dataset.inject_poison(**poison_tv)
            clean_ho_ds, poinsoned_ho_ds = dataset.get_clean_noisy_subsets('Heldout')
            dataset.set_trainset(poinsoned_ho_ds, shuffle=True)
            
                
            finetuning_cfg = None
            if 'poison' in cfg['trainer']['finetuning']:
                finetuning_cfg = cfg['trainer']['finetuning']['poison']
                finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
            else: finetuning_cfg = cfg['trainer']['finetuning']
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **finetuning_cfg,
                exp_name=experiment_name,
                exp_tags=None,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))  
            
            
        
    for idx, noise_tv in enumerate(strategy['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_n_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            if not outputs_dir/ Path(f"{cfg_name}/unpoisoned") / Path('weights/ft_weights.pth').exists():
                continue
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/unpoisoned") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            experiment_name = f"{cfg_name}/finetune_n_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            

            # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
            if noise_tv['noise_type'] == 'asymmetric':
                noise_tv['set'] = 'Heldout'
                dataset.inject_noise(**noise_tv)
                hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
                dataset.set_trainset(hs_noisy, shuffle=True)
            else:
                dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
                dataset.inject_noise(**noise_tv)
                
            finetuning_cfg = None
            if 'noise' in cfg['trainer']['finetuning']:
                finetuning_cfg = cfg['trainer']['finetuning']['noise']
                finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
            else: finetuning_cfg = cfg['trainer']['finetuning']
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **finetuning_cfg,
                exp_name=experiment_name,
                exp_tags=None,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))  
            


def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    
    dataset.reset_train_dl(shuffle=False)
    
    
    strategy = cfg['strategy']
    noise_tv = strategy['noise']['finetuning'][0]
    noise_tv['set'] = 'Heldout'
    
    dataset_clean = copy.deepcopy(dataset)
    # Inject noise first
    dataset.inject_noise(**strategy['noise']['pretraining'])
    # Inject poison next
    dataset.inject_poison(**strategy['poison']['pretraining'])
    
    # dataset.inject_noise(**noise_tv)

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
    

    
    noise_weights = OrderedDict()
    
    for noise_tv in strategy['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_n_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        noise_weights[f"N Seed {noise_tv['seed']}"] = n_weights
        
    poison_weights = OrderedDict()
    
    for poison_tv in strategy['poison']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_p_{poison_tv['rate']}_{poison_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        poison_weights[f"P Seed {poison_tv['seed']}"] = n_weights
    
    
            
    noise_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        noise_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
    
    poison_vector = OrderedDict()
    for task_name, finetuend_weights in poison_weights.items():
        poison_vector[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    
     
    noise_vectors['Average'] = TaskVector.mean(noise_vectors)
        
    
    # task_vectors['Clean'] = TaskVector(mix_weights, ft_ho_clean_weights)
    # task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    
    # task_vectors['Random Vector'] = task_vectors['Average'].generate_random_vector_with_same_layer_norms(seed=training_seed)


    
    # ft_tvs_list = list(task_vectors.values())
    # tv_names = list(task_vectors.keys())

    # task_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = ft_tvs_list[i]
    #     task_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = ft_tvs_list[j]
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



    # model.load_state_dict(mix_weights, strict=False)
    # fig_comp_pt = embedding_space_analysis.all_plot_comp(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_heldout_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_comp_pt.savefig(results_dirs['embed_plots'] / "comp_pt.png", bbox_inches="tight")
    
    # model.load_state_dict(mix_weights, strict=False)
    # # results = eval_model_on_clean_corr_splits(model, dataset, gpu)
    # # print(results)
    # test_res = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(test_res)
    # exit()
    
    
    # model.load_state_dict(mix_weights, strict=False)
    # poison_vector['P Seed 10'].apply_to(model, scaling_coef=-.91, strict=False)
    # unpoisoned_weights = model.state_dict()
    # mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # mix_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)
    
    # print(mix_test_results)
    # print(mix_train_results)
    

    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics_poison.json').exists():
    
        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)
        
        model.load_state_dict(gold_weights, strict=False)
        gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        gold_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        gold_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)
        
        # model.load_state_dict(ft_ho_clean_weights, strict=False)
        # ft_ho_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        # ft_ho_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        # ft_ho_train_results = eval_model_on_clean_corr_splits(model, None, dataset, gpu)
    
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'ho_results': mix_ho_results, 'train_results': mix_train_results}
        results_dict['Gold'] = {'test_results': gold_test_results, 'ho_results': gold_ho_results, 'train_results': gold_train_results}
        # results_dict['FT HO Clean'] = {'test_results': ft_ho_test_results, 'ho_results': ft_ho_ho_results, 'train_results': ft_ho_train_results}
        for alpha in tqdm(np.round(np.linspace(-0.05, -1.5, 30), 2)):
        
            model.load_state_dict(mix_weights, strict=False)
            poison_vector['P Seed 10'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_ho_resutls, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)

            results_dict[alpha] = {'test_results': tv_test_results, 'ho_results': tv_ho_resutls, 'train_results': tv_train_results}
        
        with open(results_dir / 'metrics_poison.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)

    else:
        with open(results_dir / "metrics_poison.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
        
    
    if 'alpha_psn' not in results_dict:
        forget_rate_thrsh = {
            'MNIST': 0.01,
            'CIFAR10': 0.01,
            'CIFAR100': 0.01
        }
        alphas = np.round(np.linspace(-0.05, -1.5, 30), 2)
        alpha_psn = 0.0
        for alpha in alphas:
            metrics = results_dict.get(alpha, None)
            if not metrics: metrics = results_dict.get(str(alpha), None)
            if not metrics: print('alpha not found', alpha)
            if round(metrics['ho_results']['ACC'], 2) <= forget_rate_thrsh[dataset.dataset_name]:
                alpha_psn = alpha
                break
        
        results_dict['alpha_psn'] = alpha_psn
        with open(results_dir / 'metrics_poison.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
        
    model.load_state_dict(mix_weights, strict=False)
    poison_vector['P Seed 10'].apply_to(model, scaling_coef=-0.95, strict=False)
    unpoisoned_weights = model.state_dict()
    
    if not Path(outputs_dir / f"{cfg_name}/unpoisoned").exists():
        outputs_dir/ Path(f"{cfg_name}/unpoisoned").mkdir()
        torch.save(unpoisoned_weights, outputs_dir/ Path(f"{cfg_name}/unpoisoned") / Path('weights/ft_weights.pth'))  
    exit()
    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics_noise.json').exists():

        model.load_state_dict(unpoisoned_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)
        
        results_dict['Unpoisoned'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
        
        if strategy['noise']['finetuning'][0]['noise_type'] == 'asymmetric':
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        else:
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        for alpha in alphas:
            
            model.load_state_dict(unpoisoned_weights, strict=False)
            noise_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_corr_splits(model, dataset, gpu)
            print(alpha, tv_test_results)
            print(alpha, tv_train_results)
            results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics_noise.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics_noise.json", "r") as json_file:
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
            state0=unpoisoned_weights,
            taskvector=noise_vectors['Average'],
            unlabeled_loader=alpha_est_support_dl,
            num_clusters=num_clusters,
            k=num_neighbor_agr_check,
            coverage_rate=coverage_rate,
            alphas=np.round(np.linspace(-0.0, -4.0, 81), 2),
            device=gpu
        )

        
        results_dict['alpha_KNN'] = alpha_kNN
        with open(results_dir / 'metrics_noise.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    

    # if 'Random Vector' not in results_dict:
    #     model.load_state_dict(mix_weights, strict=False)
    #     alpha_kNN = results_dict['alpha_KNN']
    #     task_vectors['Random Vector'].apply_to(model, scaling_coef=alpha_kNN, strict=False)
    #     random_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    #     random_train_results = eval_model_on_clean_corr_splits(model, None, dataset, gpu)
    #     results_dict['Random Vector'] = {'test_results': random_test_results, 'train_results': random_train_results}
    #     with open(results_dir / 'metrics.json' , 'w') as json_file:
    #         json.dump(results_dict, json_file, indent=4)
        
        
    
    exit()
    
    figs_alpha, fig_gold = embedding_space_analysis.pca_evolution_plot(
        model=model,
        base_weights=mix_weights,
        gold_weights=None,
        dataset=dataset_clean,
        task_vector=task_vectors['Average'],
        split='Test',
        alpha_range=np.round(np.linspace(0.0, results_dict['alpha_KNN'], 4) / 0.05) * 0.05,
        device=gpu,
        saving_dir=results_dirs['embed_plots']
    )
    
    exit()
    with open(results_dir / "metrics_seed.json", "r") as json_file:
        results_dict = json.load(json_file, object_pairs_hook=OrderedDict)


    


from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()


    parser = argparse.ArgumentParser()
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
    
    cfg_path = Path('configs/single_experiment/clip_mix_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_mix_TA").absolute()
    results_dir = Path("results/single_experiment/clip_mix_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])

        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

if __name__ == "__main__":
    main()