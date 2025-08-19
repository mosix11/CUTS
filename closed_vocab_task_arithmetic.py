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

from src.trainers import knn_eval
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


def do_knn_on_image_encoder(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    model = model_factory.create_model(cfg['model'])
    model.freeze()
    model.deactivate_projector()
    pretrained_weights = model.state_dict()
    
    for dataset_cfg in cfg['datasets']:
        if dataset_cfg['name'] != 'cifar100':
            continue
        # For knn we apply the inference transformations for both
        # training samples and test samples.
        dataset_cfg['train_transforms'] = model.get_val_transforms()
        dataset_cfg['val_transforms'] = model.get_val_transforms()
        dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        
        model.load_state_dict(pretrained_weights)
        metrics = knn_eval(
            feature_extractor=model,
            train_dl=dataset.get_train_dataloader(),
            test_dl=dataset.get_test_dataloader(),
            k=20,
            weighted=True,
            normalize=True,
            batch_size_predict=2048,
            device=trainer_utils.get_gpu_device()
        )
        
        print(f"{dataset_cfg['name']} kNN Performance with pretrained:", metrics)

        
        experiment_dir = outputs_dir / f"{cfg_name}/{dataset_cfg['name']}"
        weights_dir = experiment_dir / Path("weights")
        
        ft_weights = torch.load(weights_dir / 'ft_weights.pth', map_location=torch.device('cpu'))
        model.load_state_dict(ft_weights)
        metrics = knn_eval(
            feature_extractor=model,
            train_dl=dataset.get_train_dataloader(),
            test_dl=dataset.get_test_dataloader(),
            k=20,
            weighted=True,
            normalize=True,
            batch_size_predict=2048,
            device=trainer_utils.get_gpu_device()
        )
        
        print(f"{dataset_cfg['name']} kNN Performance with finetuned:", metrics)
        
        
def finetune_models_SCL(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    model = model_factory.create_model(cfg['model'])
    pretrained_weights = model.state_dict()
    
    for dataset_cfg in cfg['datasets']:
        dataset_cfg['train_transforms'] = model.get_train_transforms()
        dataset_cfg['val_transforms'] = model.get_val_transforms()
        dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        
        model.load_state_dict(pretrained_weights)
        # TODO: The following operation might not be needed since we
        # are loading the state dict of the full model (including MLP head)
        # from the pretrained initial weights.
        # model.activate_projector(reinitialize=True)
        
        experiment_name = f"{cfg_name}/{dataset_cfg['name']}/finetune"
        experiment_dir = outputs_dir / f"{cfg_name}/{dataset_cfg['name']}"
        
        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)
        
        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        if weights_dir.joinpath("ft_weights.pth").exists():
            continue
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['finetuning'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        
        # model.deactivate_projector()
        # metrics = knn_eval(
        #     feature_extractor=model,
        #     train_dl=dataset.get_train_dataloader(),
        #     test_dl=dataset.get_test_dataloader(),
        #     k=20,
        #     weighted=True,
        #     normalize=True,
        #     batch_size_predict=2048,
        #     device=trainer_utils.get_gpu_device()
        # )
        # print(f'KNN Evaluation on {dataset_cfg['name']}: ', metrics)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        


def linear_probe_heads(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['linear_probing']['comet_api_key'] = os.getenv("COMET_API_KEY")
    # cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")

    
    model = model_factory.create_model(cfg['model'])
    model.freeze_encoder()
    
    for head_cfg, dataset_cfg in zip(cfg['model']['heads_cfg'], cfg['datasets']):
        
        dataset_cfg['train_transforms'] = model.get_train_transforms()
        dataset_cfg['val_transforms'] = model.get_val_transforms()
        dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        
        model.activate_head(head_name=head_cfg['head_name'])
        
        experiment_name = f"{cfg_name}/{dataset_cfg['name']}/head_probe"
        experiment_dir = outputs_dir / f"{cfg_name}/{dataset_cfg['name']}"
        
        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)
        
        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        if weights_dir.joinpath("head_weights.pth").exists():
            continue
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['linear_probing'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.get_head_weights(head_name=head_cfg['head_name']), weights_dir / Path("head_weights.pth"))
        # class_names = [f"Class {i}" for i in range(num_classes)]
        # confmat = trainer.confmat("Test")
        # misc_utils.plot_confusion_matrix(
        #     cm=confmat,
        #     class_names=class_names,
        #     filepath=str(plots_dir / Path("confmat.png")),
        #     show=False,
        # )
    





def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")

    
    model = model_factory.create_model(cfg['model'])
    pretrained_weights = model.get_encoder_weights()
    
    heads_weights = OrderedDict()
    for head_cfg, dataset_cfg in zip(cfg['model']['heads_cfg'], cfg['datasets']):
        experiment_dir = outputs_dir / f"{cfg_name}/{dataset_cfg['name']}"
        head_weights = torch.load(experiment_dir / 'weights/head_weights.pth', map_location=torch.device('cpu'))
        heads_weights[head_cfg['head_name']] = head_weights
    
    model.load_heads(heads_weights)
    model.freeze_all_heads()
    
    for head_cfg, dataset_cfg in zip(cfg['model']['heads_cfg'], cfg['datasets']):
        
        dataset_cfg['train_transforms'] = model.get_train_transforms()
        dataset_cfg['val_transforms'] = model.get_val_transforms()
        dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        
        model.load_encoder(pretrained_weights)

        model.activate_head(head_name=head_cfg['head_name'])
        model.unfreeze_encoder()
        
        experiment_name = f"{cfg_name}/{dataset_cfg['name']}/finetune"
        experiment_dir = outputs_dir / f"{cfg_name}/{dataset_cfg['name']}"
        
        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)
        
        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        if weights_dir.joinpath("ft_weights.pth").exists():
            continue
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['finetuning'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.get_encoder_weights(), weights_dir / Path("ft_weights.pth"))
            
            
def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
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
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_dirs = {}
    results_dirs['cms'] = results_dir / 'confusion_mats'
    results_dirs['Ts'] = results_dir / 'transition_mats'
    results_dirs['W_norms'] = results_dir / 'weight_norms'
    results_dirs['TV_norms'] = results_dir / 'TV_norms'
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    
    parser.add_argument(
        "-k",
        "--knn",
        help="Perform kNN on the image encoder.",
        action="store_true",
    )
    
    parser.add_argument(
        "-l",
        "--linprobe",
        help="Train heads by linear probing.",
        action="store_true",
    )
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the image encoder with forzen heads.",
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
    
    cfg_path = Path('configs/single_experiment/closed_vocab_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/closed_vocab_TA").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/closed_vocab_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)


    if args.knn:
        do_knn_on_image_encoder(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    if args.linprobe:
        linear_probe_heads(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    if args.finetune:
        # finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
        finetune_models_SCL(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)