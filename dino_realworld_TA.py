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
import math
import imageio.v2 as imageio

from src.utils import embedding_space_analysis
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis



    
def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    base_model = model_factory.create_model(cfg['model'])
    
    dataset_cfg = cfg['dataset']
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()

    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)

    strategy = cfg['strategy']

    
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
            
            if strategy['finetuning_set'] == 'Test':
                dataset.set_trainset(dataset.get_testset(), shuffle=True)
                dataset.inject_noise(**noise_tv)
            elif strategy['finetuning_set'] == 'Val':
                dataset.set_trainset(dataset.get_valset(), shuffle=True)
                dataset.inject_noise(**noise_tv)
            elif strategy['finetuning_set'] == 'Val+Subset':
                valset = dataset.get_valset()
                def random_subset(ds, k, seed: int):
                    k = min(k, len(ds))
                    g = torch.Generator().manual_seed(seed)
                    idx = torch.randperm(len(ds), generator=g)[:k].tolist()
                    return Subset(ds, idx)
                valset_subset = random_subset(valset, 2000, cfg['dataset_seed'])
                dataset.set_trainset(valset_subset, shuffle=True)
                dataset.inject_noise(**noise_tv)
                
            elif strategy['finetuning_set'] == 'Train':
                dataset.inject_noise(**noise_tv)
                
            if noise_tv['noise_type'] == 'asymmetric':
                hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Train')
                dataset.set_trainset(hs_noisy, shuffle=True)
                
                
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
    
    
    
    model = model_factory.create_model(cfg['model'])
    pt_weights = copy.deepcopy(model.state_dict())
    
    dataset_cfg = cfg['dataset']
    dataset_cfg['train_transforms'] = model.get_val_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    dataset.reset_train_dl(shuffle=False)
    
    
    strategy = cfg['strategy']

    if strategy['finetuning_set'] == 'Val+Subset':
        valset = dataset.get_valset()
        def random_subset(ds, k, seed: int):
            k = min(k, len(ds))
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=g)[:k].tolist()
            return Subset(ds, idx)
        valset_subset = random_subset(valset, 2000, cfg['dataset_seed'])
        dataset.set_valset(valset_subset, shuffle=False)


    # Load weights while removing classifier weights from the state dict
    mix_weights = torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    )

    
    noise_weights = OrderedDict()
    
    for noise_tv in cfg['strategy']['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        )
        noise_weights[f"{noise_tv['noise_rate']*100:.0f}% Noise, {noise_tv['seed']} Seed"] = n_weights
        
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average'] = only_tv
    else:
        task_vectors['Average'] = TaskVector.mean(task_vectors)
  


    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():

        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        # mix_train_results, _, _ = evaluate_model(model, dataset.get_train_dataloader(), gpu)
        
        # results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
        results_dict['Mix'] = {'test_results': mix_test_results}
        print('0.0', mix_test_results)

        alphas = tqdm(np.round(np.linspace(-0.04, -2.0, 50), 3))
        for alpha in alphas:
            
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            # tv_train_results,  _, _ = evaluate_model(model, dataset.get_train_dataloader(), gpu)

            # results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
            print(alpha, tv_test_results)
            results_dict[alpha] = {'test_results': tv_test_results,}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    
    # if 'alpha_KNN' not in results_dict:
    if dataset.dataset_name == 'Clothing1M':
        coverage_rate = 1.0
    elif dataset.dataset_name == 'Food101':
        coverage_rate = 0.95

    # if strategy['noise']['finetuning'][0]['noise_type'] == 'asymmetric':
    #     if dataset.dataset_name == 'MNIST':
    #         num_clusters = 5
    #     elif dataset.dataset_name == 'CIFAR10':
    #         num_clusters = 5
    #     else: num_clusters = dataset.get_num_classes()
    # else:
    #     num_clusters = dataset.get_num_classes()
    num_clusters = dataset.get_num_classes()
    alpha_est_support_dl = dataset.get_val_dataloader()
    alpha_est_support_size = len(dataset.get_valset())
    ideal_cluster_balance = alpha_est_support_size / num_clusters
    num_neighbor_agr_check = math.floor(ideal_cluster_balance / 2)
    
    from estimate_alpha import select_alpha_by_knn_self_agreement
    alpha_kNN = select_alpha_by_knn_self_agreement(
        model=model,
        feature_extractor=model.get_feature_extractor(),
        classifier=model.get_classifier_head(),
        state0=mix_weights,
        taskvector=task_vectors['Average'],
        unlabeled_loader=alpha_est_support_dl,
        num_clusters=num_clusters,
        k=10,
        coverage_rate=coverage_rate,
        alphas=np.round(np.linspace(-0.0, -2.0, 51), 2),
        device=gpu
    )
    
    print(alpha_kNN)

    
        # results_dict['alpha_KNN'] = alpha_kNN
        # with open(results_dir / 'metrics.json' , 'w') as json_file:
        #     json.dump(results_dict, json_file, indent=4)


from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()
    
    dotenv.load_dotenv(".env")

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
        help="Apply task vectors to an already trained and finetuned experiment.",
        action="store_true",
    )
    args = parser.parse_args()

    cfg_path = Path('configs/single_experiment/dino_realworld_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/dino_realworld_TA").absolute()
    results_dir = Path("results/single_experiment/dino_realworld_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])
        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)








if __name__ == "__main__":
    main()