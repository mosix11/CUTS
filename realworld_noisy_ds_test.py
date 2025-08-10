import comet_ml
from src.datasets import dataset_factory
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
from collections import OrderedDict
import re

from helper_funcs import evaluate_model, search_optimal_coefficient, analyze_IC

    

    
def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        results[tv_name]["-1.0"] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=-1.0)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)

        results[tv_name]["-1.0"]['test_results'] = base_test_results
        
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
        
           
        
    return results

def pt_ft_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    if cfg['dataset']['name'] == 'clothing1M':
        augmentations = [
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'food101':
        augmentations = [
            transformsv2.RandomHorizontalFlip(),
        ]
    
    
    base_dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)
    base_model = model_factory.create_model(cfg['model'], num_classes)
    strategy = cfg['strategy']
    

    if not outputs_dir.joinpath(f"{cfg_name}/pretrain/weights/model_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        experiment_name = f"{cfg_name}/pretrain"

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['pretraining'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        if strategy['finetuning_set'] == 'LowLoss':
            trainer.setup_data_loaders(dataset)
            trainer.activate_low_loss_samples_buffer(
                consistency_window=5,
                consistency_threshold=0.8
            )

        # TODO set resume to False 
        results = trainer.fit(model, dataset, resume=False)
        
        # print(results)

        torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False,
        )

    

    # if not outputs_dir.joinpath(f"{cfg_name}/finetune_gold/weights/model_weights.pth").exists():
    #     dataset = copy.deepcopy(base_dataset)
    #     model = copy.deepcopy(base_model)
        
    #     base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
    #     checkpoint = torch.load(base_model_ckp_path)
    #     model.load_state_dict(checkpoint)
        
    #     clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    #     dataset.set_trainset(clean_set, shuffle=True)
            
    #     experiment_name = f"{cfg_name}/finetune_gold"
    #     experiment_dir = outputs_dir / Path(experiment_name)

    #     weights_dir = experiment_dir / Path("weights")
    #     weights_dir.mkdir(exist_ok=True, parents=True)

    #     plots_dir = experiment_dir / Path("plots")
    #     plots_dir.mkdir(exist_ok=True, parents=True)
        
        

    #     trainer = StandardTrainer(
    #         outputs_dir=outputs_dir,
    #         **cfg['trainer']['pretraining'],
    #         exp_name=experiment_name,
    #         exp_tags=None,
    #     )
        
    #     results = trainer.fit(model, dataset, resume=False)

    #     torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

    #     class_names = [f"Class {i}" for i in range(num_classes)]
    #     confmat = trainer.confmat("Test")
    #     misc_utils.plot_confusion_matrix(
    #         cm=confmat,
    #         class_names=class_names,
    #         filepath=str(plots_dir / Path("confmat.png")),
    #         show=False,
    #     )
        

    
    for idx, noise_tv in enumerate(strategy['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
            checkpoint = torch.load(base_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout':
                dataset.set_trainset(dataset.get_valset(), shuffle=True)
                dataset.inject_noise(**noise_tv)
                    
            elif strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{strategy['percentage']:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.subset_set(set='Train', indices=all_easy_samples)
                
                dataset.inject_noise(**noise_tv)
                
            elif strategy['finetuning_set'] == 'HighLoss':
                pass
            
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['finetuning'],
                exp_name=experiment_name,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            print(results)

            torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

            class_names = [f"Class {i}" for i in range(num_classes)]
            confmat = trainer.confmat("Test")
            misc_utils.plot_confusion_matrix(
                cm=confmat,
                class_names=class_names,
                filepath=str(plots_dir / Path("confmat.png")),
                show=False
            )
      
      
def save_samples_loss_ranked(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    
    model = model_factory.create_model(cfg['model'], num_classes)
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    base_expr_dir = outputs_dir / cfg_name
    pretrain_dir = base_expr_dir / 'pretrain'
    
    pretrain_weights = torch.load(pretrain_dir / 'weights/model_weights.pth', map_location=cpu)
    
    model.load_state_dict(pretrain_weights)
    dataset.set_trainset(dataset.get_trainset(), shuffle=False)
    trainloader = dataset.get_train_dataloader()
    
    
    
            
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
    
    base_expr_dir = outputs_dir / cfg_name
    pretrain_dir = base_expr_dir / 'pretrain'
    finetune_dirs = OrderedDict()
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        ft_expr_dir = base_expr_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        finetune_dirs[f"{noise_tv['noise_rate']}_{noise_tv['seed']}"] = ft_expr_dir
        
    pretrain_weights = torch.load(pretrain_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_dirs.items():
        finetune_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
    
    
    weight_norm_analysis.plot_abs_weight_norms_compare(
        state_dicts={
            'Pretrain': pretrain_weights,
            # 'FT Gold': ft_gold_wieghts,
            'FT Noise': next(iter(finetune_weights.items()))[1]
            },
        include_bias_and_norm=False,
        max_groups=40,
        overall_bins=200,
        layer_bins=200,
        logy=False,
        saving_path=results_dir / 'abs_weight_norm.png'
    )
    
    finetune_tvs = OrderedDict()
    
    for ft_expr, ft_weight in finetune_weights.items():
        finetune_tvs[f"{float(ft_expr.split('_')[0])*100:.0f}% Noise, {ft_expr.split('_')[1]} Seed"] = TaskVector(pretrain_weights, ft_weight)
    finetune_tvs['Average TV'] = TaskVector.mean(finetune_tvs)
    finetune_tvs['Average TV Pruned 0.4'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.4)
    finetune_tvs['Average TV Pruned 0.6'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.6)
    finetune_tvs['Average TV Pruned 0.8'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.8)
    finetune_tvs['Average TV Pruned 0.9'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.9)
    finetune_tvs['Average TV Pruned 0.95'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.95)
    finetune_tvs['Average TV Pruned 0.99'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.99)
    finetune_tvs['Random Vector'] = finetune_tvs['Average TV'].generate_random_vector_with_same_layer_norms(seed=11)
    
    # finetune_tvs.pop('100% Noise, 12 Seed')
    # finetune_tvs.pop('100% Noise, 10 Seed')
    # finetune_tvs.pop('100% Noise, 15 Seed')
    # finetune_tvs.pop('100% Noise, 20 Seed')
    # finetune_tvs.pop('100% Noise, 30 Seed')

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
            'Average TV Pruned 0.99': finetune_tvs['Average TV Pruned 0.99'].vector
            },
        include_bias_and_norm=False,
        max_groups=40,
        overall_bins=200,
        layer_bins=200,
        logy=False,
        saving_path=results_dir / 'abs_weight_norm_TV.png'
    )
    
    weight_norm_analysis.plot_l2_weight_norms_compare(
        state_dicts={
            'Average TV': finetune_tvs['Average TV'].vector,
            'Average TV Pruned 0.99': finetune_tvs['Average TV Pruned 0.99'].vector
            },
        include_bias_and_norm=False,
        max_groups=40,
        overall_bins=200,
        layer_bins=200,
        logy=False,
        saving_path=results_dir / 'L2_weight_norm_TV.png'
    )
    
    base_model.load_state_dict(pretrain_weights)
    analyze_IC(
        base_model,
        dataset.get_num_classes(),
        dataset.get_val_dataloader(),
        gpu,
        dataset.get_class_names()
        )
    
    exit()

    
    base_model.load_state_dict(pretrain_weights)
    pt_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)

    
    results_dict = OrderedDict()
    
    results_dict['Pretrain'] = {'test_results': pt_test_results}

    results_dict = eval_model_on_tvs(base_model, finetune_tvs, results_dict, cfg, dataset, num_classes, gpu)
    # results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names[-1:], ft_tvs_list[-1:])), results_dict, cfg, dataset, num_classes, gpu)
    
    print(results_dict)
        
        
    with open(results_dir / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)

    # generate_latex_table_from_results(results_dict, results_dir / 'results_tex.txt')
    

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
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
    
    cfg_path = Path('configs/single_experiment/realworld_pretrain_on_noisy') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/realworld_pretrain_on_noisy").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/realworld_pretrain_on_noisy").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    else:
        pt_ft_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)