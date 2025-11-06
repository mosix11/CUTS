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
from helper_funcs import evaluate_model, eval_model_on_clean_corrupted_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis

    
    
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
    base_dataset.inject_poison(**strategy['poison']['pretraining'])

            

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
        
        
    # Finetune on the set we use for poison vectors but with uncorrupted labels.
    if not outputs_dir.joinpath(f"{cfg_name}/finetune_clean/weights/ft_weights.pth").exists() and strategy['finetuning_set'] == 'Heldout':
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
        checkpoint = torch.load(mix_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        
        p_tv = copy.deepcopy(strategy['poison']['finetuning'][0])
        
        p_tv['set'] = 'Heldout'
        p_tv['rate'] = 0.0
        dataset.inject_poison(**p_tv)
        dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
            
        
        experiment_name = f"{cfg_name}/finetune_clean"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        finetuning_cfg = None
        if 'heldout' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['heldout']
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
        
        
    # # Gradient Ascent Baseline
    # if not outputs_dir.joinpath(f"{cfg_name}/gradient_ascent/weights/ft_weights.pth").exists():
    #     dataset = copy.deepcopy(base_dataset)
    #     model = copy.deepcopy(base_model)
        
    #     mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
    #     checkpoint = torch.load(mix_model_ckp_path)
    #     model.load_state_dict(checkpoint)
        
        
    #     dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
        
    #     experiment_name = f"{cfg_name}/gradient_ascent"
    #     experiment_dir = outputs_dir / Path(experiment_name)

    #     weights_dir = experiment_dir / Path("weights")
    #     weights_dir.mkdir(exist_ok=True, parents=True)

    #     plots_dir = experiment_dir / Path("plots")
    #     plots_dir.mkdir(exist_ok=True, parents=True)
        
    #     if strategy['finetuning_set'] == 'Heldout':
    #         dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
    #         dataset.inject_noise(**strategy['noise']['finetuning'][0])
            
    #     finetuning_cfg = None
    #     if 'gradient_ascent' in cfg['trainer']['finetuning']:
    #         finetuning_cfg = cfg['trainer']['finetuning']['gradient_ascent']
    #         finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
    #     else: finetuning_cfg = cfg['trainer']['finetuning']
        
    #     trainer = GradientAscentTrainer(
    #         outputs_dir=outputs_dir,
    #         **finetuning_cfg,
    #         exp_name=experiment_name,
    #         exp_tags=None,
    #     )
        
    #     results = trainer.fit(model, dataset, resume=False)
    #     torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
    for idx, poison_tv in enumerate(strategy['poison']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{poison_tv['rate']}_{poison_tv['seed']}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
                    
            experiment_name = f"{cfg_name}/finetune_{poison_tv['rate']}_{poison_tv['seed']}"
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
    
    dataset_clean = copy.deepcopy(dataset)
    
    strategy = cfg['strategy']
    dataset.inject_poison(**strategy['poison']['pretraining'])
    
    poison_tv_cfg = strategy['poison']['finetuning'][0]
    poison_tv_cfg['set'] = 'Heldout'
    dataset.inject_poison(**poison_tv_cfg)
    # Exclude clean samples from target class
    clean_ho_ds, poinsoned_ho_ds = dataset.get_clean_noisy_subsets('Heldout')
    dataset.set_heldoutset(poinsoned_ho_ds)
    



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
    
    # ft_gradient_ascent_weights = OrderedDict(
    # (k, v) for k, v in torch.load(
    #     outputs_dir.joinpath(f"gradient_ascent/weights/ft_weights.pth"),
    #     map_location='cpu'
    # ).items() if "classifier_heads" not in k)
    
    poison_weights = OrderedDict()
    
    for poison_tv in cfg['strategy']['poison']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{poison_tv['rate']}_{poison_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        poison_weights[f"{poison_tv['rate']*100:.0f}% Noise, {poison_tv['seed']} Seed"] = n_weights
        
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in poison_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average'] = only_tv
    else:
        task_vectors['Average'] = TaskVector.mean(task_vectors)

    task_vectors['CF'] = TaskVector(mix_weights, ft_ho_clean_weights)
    task_vectors['Random Vector'] = task_vectors['Average'].generate_random_vector_with_same_layer_norms(seed=20)
    task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    # task_vectors['Gold'] = TaskVector(pt_weights, gold_weights)

    # with open(results_dir / "metrics.json", "r") as json_file:
    #     results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
    # estimated_poison_vector =  task_vectors['Average'] * (-1 * results_dict['alpha_psn'])
    # estimated_clean_vector = task_vectors['Mix'] - estimated_poison_vector
    # task_vectors['Clean'] = estimated_clean_vector
    
    ft_tvs_list = list(task_vectors.values())
    tv_names = list(task_vectors.keys())

    # TV_norms = OrderedDict()
    # for name, tv in task_vectors.items():
    #     TV_norms[name] = tv.norm().item()
    # with open(results_dirs['TV_norms'] / 'norms.json' , 'w') as json_file:
    #     json.dump(TV_norms, json_file, indent=4)
        
    # task_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = ft_tvs_list[i]
    #     task_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = ft_tvs_list[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         task_sim[i].append(cos_sim)
    # task_sim = np.array(task_sim)
    
    # with open(results_dirs['cms'] / "tv_sim.pkl", "wb") as f:
    #     pickle.dump(task_sim, f)
    
    
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
    #     figsize=(8, 6),
    #     tick_label_font_size=10,
    #     filepath=results_dir / 'task_similarities.png',
    #     show=False
    # )
    
    

    
    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():
    
        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)
        
        model.load_state_dict(gold_weights, strict=False)
        gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        gold_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        gold_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)
        
        model.load_state_dict(ft_ho_clean_weights, strict=False)
        ft_ho_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        ft_ho_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        ft_ho_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)
    
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'ho_results': mix_ho_results, 'train_results': mix_train_results}
        results_dict['Gold'] = {'test_results': gold_test_results, 'ho_results': gold_ho_results, 'train_results': gold_train_results}
        results_dict['FT HO Clean'] = {'test_results': ft_ho_test_results, 'ho_results': ft_ho_ho_results, 'train_results': ft_ho_train_results}
        for alpha in tqdm(np.round(np.linspace(-0.05, -1.5, 30), 2)):
        
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_ho_resutls, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)

            results_dict[alpha] = {'test_results': tv_test_results, 'ho_results': tv_ho_resutls, 'train_results': tv_train_results}
        
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)

    else:
        with open(results_dir / "metrics.json", "r") as json_file:
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
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
 
    if 'Random Vector' not in results_dict:
        model.load_state_dict(mix_weights, strict=False)
        alpha_psn = results_dict['alpha_psn']
        task_vectors['Random Vector'].apply_to(model, scaling_coef=alpha_psn, strict=False)
        random_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        random_ho_resutls, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        random_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)
        results_dict['Random Vector'] = {'test_results': random_test_results, 'ho_results':random_ho_resutls, 'train_results': random_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
            
    
    
    
    tmp_dataset = copy.deepcopy(dataset_clean)
    tmp_dataset.inject_poison(**poison_tv_cfg)
    # Exclude clean samples from target class
    # clean_ho_ds, poinsoned_ho_ds = dataset.get_clean_noisy_subsets('Heldout')
    # dataset.set_heldoutset(poinsoned_ho_ds)
    ho_set = tmp_dataset.get_heldoutset()
    tmp_dataset.switch_labels_to_clean(ho_set)
    tmp_dataset.set_heldoutset(ho_set)
    figs_alpha, fig_gold = embedding_space_analysis.pca_evolution_plot(
        model=model,
        base_weights=mix_weights,
        gold_weights=gold_weights,
        dataset=tmp_dataset,
        task_vector=task_vectors['Average'],
        split='Heldout',
        # alpha_range=np.round(np.linspace(0.0, results_dict['alpha_psn'], 4) / 0.05) * 0.05,
        alpha_range=np.linspace(0.0, results_dict['alpha_psn'], 60),
        align_on='points',
        device=gpu,
        saving_dir=results_dirs['embed_plots']
    )  
    
    exit()

    # Weight Space Disentanglemet Analysis
    estimated_poison_vector =  task_vectors['Average'] * (-1 * results_dict['alpha_psn'])
    estimated_clean_vector = task_vectors['Mix'] - estimated_poison_vector
    
    subset_size  = 1024
    def random_subset(ds, k, seed: int):
        k = min(k, len(ds))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:k].tolist()
        return Subset(ds, idx)
    
    
    poisoned_support = dataset.get_heldoutset()
    # clean_support = dataset_clean.get_heldoutset()
    # non_target_indices = []
    # for smp_idx, sample in enumerate(clean_support):
    #     if sample[1] != 0:
    #         non_target_indices.append(smp_idx)
    # clean_support = Subset(clean_support, non_target_indices)
    # print(len(poisoned_support), len(clean_support))
    clean_support = random_subset(dataset.get_testset(), k=subset_size, seed=dataset_seed)


    # model.load_state_dict(pt_weights, strict=False)
    # pt_test_results, _, _= evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(pt_test_results)

    # model.load_state_dict(pt_weights, strict=False)
    # estimated_clean_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # clean_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(clean_test_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # estimated_poison_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # poison_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(poison_test_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # pt_ho_results, _, _= evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    # print(pt_ho_results)

    # model.load_state_dict(pt_weights, strict=False)
    # estimated_clean_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # clean_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    # print(clean_ho_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # estimated_poison_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # poison_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    # print(poison_ho_results)
    
    
    # model.load_state_dict(pt_weights, strict=False)
    # noise_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # tv_hot_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    # print(tv_hot_results)
    # tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(tv_test_results)
    # tv_train_results = eval_model_on_clean_corrupted_splits(model, None, dataset, gpu)
    # print(tv_train_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_analysis(
    #     model=model,
    #     taskvector1=estimated_clean_vector,
    #     support_tv1=clean_support,
    #     taskvector2=estimated_poison_vector,
    #     support_tv2=poisoned_support,
    #     alhpa_range=(0.0, 2.0),
    #     step=0.1,
    #     batch_size=512,
    #     device=gpu
    # )
    # with open(results_dir / "WD2.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)
    
    from alignemnt_score import compute_task_vector_alignment
    
    alingment_score = compute_task_vector_alignment(
        model=model,
        clean_tv=estimated_clean_vector,
        corruption_tv=estimated_poison_vector,
        testset_tv1=clean_support,
        testset_tv2=poisoned_support,
        dataset_name=dataset.dataset_name,
        corruption_type='pois',
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
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the image encoder with forzen heads on poisoned datasets.",
        action="store_true",
    )
    
    parser.add_argument(
        "-t",
        "--tv",
        help="Apply task vectors to an already trained and finetuned experiment.",
        action="store_true",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/clip_poison_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_poison_TA").absolute()
    results_dir = Path("results/single_experiment/clip_poison_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])
        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)


if __name__ == "__main__":
    main()