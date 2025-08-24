import comet_ml
from src.datasets import dataset_factory, dataset_wrappers
from src.models import model_factory, TaskVector, weight_norm_analysis
from src.trainers import StandardTrainer, utils as trainer_utils
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

from src.trainers import umap_plot
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize

def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        results[tv_name]["-1.0"] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=-1.0)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
        base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name]["-1.0"]['test_results'] = base_test_results
        results[tv_name]["-1.0"]['train_results'] = base_train_split_results
        
        base_model = copy.deepcopy(model)

        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=tv,
            search_range=(-2.0, 0.0),
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



def finetune_models_gt(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    dataset_cfg = cfg['datasets'][0]
    noise_cfg = dataset_cfg.pop('noise_cfg')
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    
    base_dataset.inject_noise(**noise_cfg)
    
    
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
        
        
        
    if not outputs_dir.joinpath(f"{cfg_name}/noise/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)  
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(noisy_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/noise"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
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
    base_dataset.inject_noise(**strategy['noise']['pretraining'])

    
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
        
        
    # Finetune on the set we use for noise vectors but with uncorrupted labels.
    if not outputs_dir.joinpath(f"{cfg_name}/finetune_clean/weights/ft_weights.pth").exists() and strategy['finetuning_set'] == 'Heldout':
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
        checkpoint = torch.load(mix_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        
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
        
        
    for idx, noise_tv in enumerate(strategy['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout':
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
            


def apply_tv_gt(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
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
    noise_cfg = dataset_cfg.pop('noise_cfg')
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: dataset.get_class_names()} 
    model = model_factory.create_model(cfg['model'])
    model.freeze_all_heads()
    
    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
    dataset_cfg['train_transforms'] = model.get_train_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    dataset.inject_noise(**noise_cfg)

    ft_weights = OrderedDict()


    # Load finetuned weights while removing classifier weights from the state dict
    ft_weights['mix'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_weights['clean'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_weights['noise'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/noise/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in ft_weights.items():
        task_vectors[task_name] = TaskVector(pt_weights, finetuend_weights)

    sum_clean_noise_TV = TaskVector.sum([task_vectors['clean'] + task_vectors['noise']])
    avg_clean_noise_TV = TaskVector.mean([task_vectors['clean'] + task_vectors['noise']])
    goal_TV = task_vectors['mix'] - (task_vectors['noise'])
    
    
    
    ft_tvs_list = list(task_vectors.values())
    tv_names = list(task_vectors.keys())
    ft_tvs_list.extend([sum_clean_noise_TV, avg_clean_noise_TV, goal_TV])
    tv_names.extend(['Sum TV', 'Avg TV', 'Goal TV'])
    
    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
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


    
    model.load_state_dict(pt_weights, strict=False)
    pt_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    pt_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(ft_weights['mix'], strict=False)
    mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(ft_weights['noise'], strict=False)
    noise_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    noise_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(ft_weights['clean'], strict=False)
    clean_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    clean_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(pt_weights, strict=False)
    
    results_dict = OrderedDict()
    
    results_dict['Pretrain'] = {'test_results': pt_test_results, 'train_results': pt_train_results}
    results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
    results_dict['Clean'] = {'test_results': clean_test_results, 'train_results': clean_train_results}
    results_dict['Noise'] = {'test_results': noise_test_results, 'train_results': noise_train_results}
    

    for alpha in tqdm(np.linspace(-0.1, -1.0, 10)):
    
        model.load_state_dict(ft_weights['mix'], strict=False)
        task_vectors['noise'].apply_to(model, scaling_coef=alpha, strict=False)
        tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

        results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    
    with open(results_dirs['metrics'] / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    # with open(results_dir / 'tv_metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)


def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    
    dataset_cfg['train_transforms'] = model.get_train_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
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
        
    task_vectors['Average TV Pruned 0.4'] = task_vectors['Average TV'].prune_small_weights(rate=0.4)
    task_vectors['Average TV Pruned 0.6'] = task_vectors['Average TV'].prune_small_weights(rate=0.6)
    task_vectors['Average TV Pruned 0.8'] = task_vectors['Average TV'].prune_small_weights(rate=0.8)
    task_vectors['Average TV Pruned 0.9'] = task_vectors['Average TV'].prune_small_weights(rate=0.9)
    task_vectors['Average TV Pruned 0.95'] = task_vectors['Average TV'].prune_small_weights(rate=0.95)
    task_vectors['Average TV Pruned 0.99'] = task_vectors['Average TV'].prune_small_weights(rate=0.99)
    task_vectors['Random Vector'] = task_vectors['Average TV'].generate_random_vector_with_same_layer_norms(seed=11)

    
    
    
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

    
    # model.load_state_dict(mix_weights, strict=False)
    # umap_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset.get_train_dataloader(),
    #     device=gpu,
    # )
    
    # task_vectors['Average TV'].apply_to(model, scaling_coef=-1.0, strict=False)
    # umap_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset.get_train_dataloader(),
    #     device=gpu,
    # )
    
    # exit()
    
    model.load_state_dict(mix_weights, strict=False)
    mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    
    model.load_state_dict(gold_weights, strict=False)
    gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    gold_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(pt_weights, strict=False)
    
    results_dict = OrderedDict()
    
    results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
    results_dict['Gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
    
    
    # results_dict = OrderedDict()
    for alpha in tqdm(np.linspace(-0.1, -1.0, 10)):
    
        model.load_state_dict(mix_weights, strict=False)
        task_vectors['Average TV'].apply_to(model, scaling_coef=alpha, strict=False)
        tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

        results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    
    with open(results_dir / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    # with open(results_dir / 'tv_metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)
    
def apply_tvs(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
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

    
    
    
    base_expr_dir = outputs_dir / cfg_name
    gold_dir = base_expr_dir / 'gold'
    pretrain_dir = base_expr_dir / 'pretrain'
    ft_gold_dir = base_expr_dir / 'finetune_gold'
    ft_gt_noise_dir = base_expr_dir / 'finetune_gt_noise'
    finetune_dirs = OrderedDict()
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        ft_expr_dir = base_expr_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        finetune_dirs[f"{noise_tv['noise_rate']}_{noise_tv['seed']}"] = ft_expr_dir
        
    gold_weights = torch.load(gold_dir / 'weights/model_weights.pth', map_location=cpu)
    pretrain_weights = torch.load(pretrain_dir / 'weights/model_weights.pth', map_location=cpu)
    ft_gold_wieghts = torch.load(ft_gold_dir / 'weights/model_weights.pth', map_location=cpu)
    ft_gt_noise_weights = torch.load(ft_gt_noise_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_dirs.items():
        finetune_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
 
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Pretrain': pretrain_weights,
            'Gold': gold_weights,
            'FT Noise': next(iter(finetune_weights.items()))[1]
            },
        saving_path=results_dirs['W_norms'] / 'L1_pt_gold_ftnoise.png'
    )
    
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Pretrain': pretrain_weights,
            'FT Gold': ft_gold_wieghts,
            'FT Noise': next(iter(finetune_weights.items()))[1]
            },
        saving_path=results_dirs['W_norms'] / 'L1_pt_ftgold_ftnoise.png'
    )
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'FT Gold': ft_gold_wieghts,
            'FT Noise': next(iter(finetune_weights.items()))[1]
            },
        saving_path=results_dirs['W_norms'] / 'L1_ftgold_ftnoise.png'
    )
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Gold': gold_weights,
            'Pretrain': pretrain_weights,
            'FT Gold': ft_gold_wieghts,
            },
        saving_path=results_dirs['W_norms'] / 'L1_pt_gold_ftgold.png'
    )
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Pretrain': pretrain_weights,
            'FT Gold': ft_gold_wieghts,
            'FT Noise': next(iter(finetune_weights.items()))[1],
            'FT GT Noise': ft_gt_noise_weights
            },
        saving_path=results_dirs['W_norms'] / 'L1_pt_ftgold_ftnoise_gtnoise.png'
    )
    
    
    ft_gold_tv = TaskVector(pretrain_weights, ft_gold_wieghts)
    ft_gt_noise_tv = TaskVector(pretrain_weights, ft_gt_noise_weights)

    finetune_tvs = OrderedDict()
    
    for ft_expr, ft_weight in finetune_weights.items():
        finetune_tvs[f"{float(ft_expr.split('_')[0])*100:.0f}% Noise, {ft_expr.split('_')[1]} Seed"] = TaskVector(pretrain_weights, ft_weight)
        
    if len(finetune_tvs) == 1:
        finetune_tvs['Average TV'] = list(finetune_tvs.items())[0][1]
        finetune_tvs.popitem(last=False)
    else:
        finetune_tvs['Average TV'] = TaskVector.mean(finetune_tvs)
    finetune_tvs['Average TV Pruned 0.4'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.4)
    finetune_tvs['Average TV Pruned 0.6'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.6)
    finetune_tvs['Average TV Pruned 0.8'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.8)
    finetune_tvs['Average TV Pruned 0.9'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.9)
    finetune_tvs['Average TV Pruned 0.95'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.95)
    finetune_tvs['Average TV Pruned 0.99'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.99)
    finetune_tvs['Random Vector'] = finetune_tvs['Average TV'].generate_random_vector_with_same_layer_norms(seed=11)
    finetune_tvs['Gold'] = ft_gold_tv
    finetune_tvs['Ground Truth Noise'] = ft_gt_noise_tv
    finetune_tvs.move_to_end('Ground Truth Noise', last=False)
    finetune_tvs.move_to_end('Gold', last=False)
    
    

    ft_tvs_list = list(finetune_tvs.values())
    print(finetune_tvs.keys())
    tv_names = list(finetune_tvs.keys())
    
    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
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
    
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Average TV': finetune_tvs['Average TV'].vector,
            'Gold TV': finetune_tvs['Gold'].vector,
            # 'Average TV Pruned 0.8': finetune_tvs['Average TV Pruned 0.8'].vector
            },
        saving_path=results_dirs['TV_norms'] / 'L1_norms.png'
    )
    
    
    weight_norm_analysis.plot_l2_weight_norms_compare(
        state_dicts={
            'Average TV': finetune_tvs['Average TV'].vector,
            'Gold TV': finetune_tvs['Gold'].vector,
            # 'Average TV Pruned 0.8': finetune_tvs['Average TV Pruned 0.8'].vector
            },
        saving_path=results_dirs['TV_norms'] / 'L2_norms.png'
    )
    
    
    base_model.load_state_dict(pretrain_weights)

    cm_pt = get_confusion_matrix(
        base_model,
        dataset.get_num_classes(),
        dataset.get_heldout_dataloader(),
        gpu
    )
    
    # T = estimate_T_from_confusion(cm_pt, alpha=0.01, lam=0.1)
    T= None
    
    misc_utils.plot_confusion_matrix(
        title='Noise Transition Matrix',
        cm=T,
        class_names=dataset.get_class_names(),
        color_map='vlag',
        color_bar=True,
        # vmin= 0.0,
        # vmax= 1.0,
        x_label='Classes',
        y_label='Classes',
        tick_label_font_size=6,
        filepath=results_dirs['Ts'] / 'transition_matrix.png',
        show=False
    )
    
    
    
    
    misc_utils.plot_confusion_matrix(
        title='Normalized Confusion Matrix',
        cm=row_normalize(cm_pt),
        class_names=dataset.get_class_names(),
        color_map='vlag',
        color_bar=True,
        # vmin= 0.0,
        # vmax= 1.0,
        x_label='Classes',
        y_label='Classes',
        tick_label_font_size=6,
        filepath=results_dirs['cms'] / 'pretrained_normalized.png',
        show=False
    )
    
    
    ft_model = copy.deepcopy(base_model)
    ft_model.load_state_dict(next(iter(finetune_weights.items()))[1])
    cm_ft = get_confusion_matrix(
        ft_model,
        dataset.get_num_classes(),
        dataset.get_heldout_dataloader(),
        gpu
    )
    
    misc_utils.plot_confusion_matrix(
        title='Normalized Confusion Matrix',
        cm=row_normalize(cm_ft),
        class_names=dataset.get_class_names(),
        color_map='vlag',
        color_bar=True,
        # vmin= 0.0,
        # vmax= 1.0,
        x_label='Classes',
        y_label='Classes',
        tick_label_font_size=6,
        filepath=results_dirs['cms'] / 'ft_noise_normalized.png',
        show=False
    )
    
    finetune_tvs['Average TV'].apply_to(base_model, scaling_coef=-1.0)
    cm_ng = get_confusion_matrix(
        base_model,
        dataset.get_num_classes(),
        dataset.get_heldout_dataloader(),
        gpu
    )
    
    misc_utils.plot_confusion_matrix(
        title='Normalized Confusion Matrix',
        cm=row_normalize(cm_ng),
        class_names=dataset.get_class_names(),
        color_map='vlag',
        color_bar=True,
        # vmin= 0.0,
        # vmax= 1.0,
        x_label='Classes',
        y_label='Classes',
        tick_label_font_size=6,
        filepath=results_dirs['cms'] / 'negated_normalized.png',
        show=False
    )
        

    # rank_dict = OrderedDict()
    # for tv_name, tv in finetune_tvs.items():
    #     rank_dict[tv_name] = tv.get_layer_rank()
        
    # with open(results_dir / 'ranks.json' , 'w') as json_file:
    #     json.dump(rank_dict, json_file, indent=4)
    
    
    # for i in range(len(ft_tvs_list)):
    #     if i == 0:
    #         print('passing ft gold from low rank approximation')
    #         continue
    #     else:
    #         ftsv = ft_tvs_list[i].compute_SVD_for_each_layer(k=0.1)
        
    #         ft_tvs_list[i].apply_SVD_to_TV(ftsv)
    
    
    # task_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = ft_tvs_list[i]
    #     task_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = ft_tvs_list[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         task_sim[i].append(cos_sim)
    # task_sim = np.array(task_sim)
    
    # misc_utils.plot_confusion_matrix(cm=task_sim, class_names=class_names, filepath=None, show=True)
    
    # for ft_name, ft_tv in finetune_tvs.items():
    #     best_coef, best_results, best_cm = search_optimal_coefficient(
    #         base_model=base_model,
    #         task_vector=ft_tv,
    #         search_range=(-1.5, 0.0),
    #         dataset=dataset,
    #         num_classes=num_classes,
    #         device=gpu
    #     )
    #     print(f"Best scaling coefficient for {ft_name} = {best_coef}")
    #     print(f"Metrics of the negated model is {best_results}")
            
    
    

        
    # TSV = TaskVector.TSV_extract_common_direction(finetune_tvs, k=0.3)
    # TSV.apply_to(base_model, scaling_coef=1.0)
        
    # for ft_name, ft_tv in finetune_tvs.items():
    #     best_coef, best_results, best_cm = search_optimal_coefficient(
    #         base_model=base_model,
    #         task_vector=ft_tv,
    #         search_range=(-1.5, 0.0),
    #         dataset=dataset,
    #         num_classes=num_classes,
    #         device=gpu
    #     )
    #     print(f"Best scaling coefficient for {ft_name} = {best_coef}")
    #     print(f"Metrics of the negated model is {best_results}")
    
    # test_tv = (ft_tvs_list[2] + ft_tvs_list[3]) * 0.5
    
    # ordered_dict = OrderedDict(zip(tv_names[1:], ft_tvs_list[1:]))
    
    # best_coef, best_results, best_cm = search_optimal_coefficient(
    #     base_model=base_model,
    #     # task_vector=test_tv,
    #     task_vector=ft_tvs_list[4],
    #     search_range=(-3.0, 0.0),
    #     dataset=dataset,
    #     num_classes=num_classes,
    #     device=gpu
    # )
    
    # print(f"Best scaling coefficient for TV = {best_coef}")
    # print(f"Metrics of the negated model is {best_results}")
    
    # before_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print('Performance before TV:', before_tv_metrics)
    # base_model.to(cpu)
    # # test_tv.apply_to(base_model, scaling_coef=best_coef)
    # ft_tvs_list[4].apply_to(base_model, scaling_coef=best_coef)
    
    # after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print('Performance after TV:', after_tv_metrics)
    
    # base_model.load_state_dict(next(iter(finetune_weights.items()))[1])
    # temp_res = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print(temp_res)
    # exit()
    
    base_model.load_state_dict(pretrain_weights)
    pt_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    pt_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(gold_weights)
    gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    gold_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(ft_gold_wieghts)
    ft_gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    ft_gold_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(pretrain_weights)
    base_model.to(cpu)
    
    results_dict = OrderedDict()
    
    results_dict['Pretrain'] = {'test_results': pt_test_results, 'train_results': pt_train_results}
    results_dict['Gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
    results_dict['Finetune Gold'] = {'test_results': ft_gold_test_results, 'train_results': ft_gold_train_results}
    results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names[1:], ft_tvs_list[1:])), results_dict, cfg, dataset, num_classes, gpu)
    # results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names, ft_tvs_list)), results_dict, cfg, dataset, num_classes, gpu)
    
    
    # print(results_dict)
        
        
    with open(results_dirs['metrics'] / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)

    




    
    


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
    
    
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the image encoder with forzen heads on noisy datasets.",
        action="store_true",
    )
    
    parser.add_argument(
        "-g",
        "--groundtruth",
        help="Finetune the image encoder with forzen heads on noisy datasets using ground truth noise.",
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
    
    cfg_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_noise_TA").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/clip_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

        
    if args.finetune and args.groundtruth:
        finetune_models_gt(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv and args.groundtruth:
        apply_tv_gt(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)