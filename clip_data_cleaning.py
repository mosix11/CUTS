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


from src.utils import embedding_space_analysis
from helper_funcs import evaluate_model, eval_model_on_clean_corrupted_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis



    
def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    dataset_cfg = cfg['dataset']
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
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
            

            dataset.set_trainset(dataset.get_testset(), shuffle=True)
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
    
    
    dataset_cfg = cfg['dataset']
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


    # Load weights while removing classifier weights from the state dict
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    
    noise_weights = OrderedDict()
    
    for noise_tv in strategy['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
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
        


    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():

        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_train_results, _, _ = evaluate_model(model, dataset.get_train_dataloader(), gpu)
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}


        alphas = tqdm(np.round(np.linspace(-0.02, -1.0, 50), 3))
        for alpha in alphas:
            
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_train_results,  _, _ = evaluate_model(model, dataset.get_train_dataloader(), gpu)

            results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    # if 'alpha_KNN' not in results_dict:        
    #     from estimate_alpha import select_alpha_by_knn_self_agreement
    #     alpha_kNN = select_alpha_by_knn_self_agreement(
    #         model=model,
    #         feature_extractor=model.get_feature_extractor(),
    #         classifier=model.get_active_head(),
    #         state0=mix_weights,
    #         taskvector=task_vectors['Average'],
    #         unlabeled_loader=dataset_clean.get_heldout_dataloader(),
    #         # K=dataset.get_num_classes(),
    #         alphas=np.round(np.linspace(-0.02, -1.0, 50), 3),
    #         device=gpu
    #     )

    #     results_dict['alpha_KNN'] = alpha_kNN
    #     with open(results_dir / 'metrics.json' , 'w') as json_file:
    #         json.dump(results_dict, json_file, indent=4)

        
    
    # figs_alpha, fig_gold = embedding_space_analysis.pca_evolution_plot(
    #     model=model,
    #     base_weights=mix_weights,
    #     gold_weights=None,
    #     dataset=dataset_clean,
    #     task_vector=task_vectors['Average'],
    #     split='Test',
    #     alpha_range=np.round(np.linspace(0.0, results_dict['alpha_KNN'], 4) / 0.05) * 0.05,
    #     device=gpu,
    #     saving_dir=results_dirs['embed_plots']
    # )
    





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
        "-g",
        "--groundtruth",
        help="Finetune the image encoder with forzen heads on noisy datasets using ground truth noise.",
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
    
    cfg_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_noise_TA").absolute()
    results_dir = Path("results/single_experiment/clip_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])

        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

if __name__ == "__main__":
    main()