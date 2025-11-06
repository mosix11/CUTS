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
from helper_funcs import evaluate_model, eval_model_on_clean_corrupted_splits


def apply_tv(outputs_dir: Path, results_dir: Path, cfg1: dict, cfg2: dict, cfg1_name:str, cfg2_name:str):
    training_seed = cfg1['training_seed']
    dataset_seed = cfg1['dataset_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    outputs1_dir = outputs_dir / cfg1_name
    results1_dir = results_dir / cfg1_name

    outputs2_dir = outputs_dir / cfg2_name
    results2_dir = results_dir / cfg2_name
    
    
    dataset1_cfg = cfg1['datasets'][0]
    dataset1, num_classes1 = dataset_factory.create_dataset(dataset1_cfg)
    
    dataset2_cfg = cfg2['datasets'][0]
    dataset2, num_classes2 = dataset_factory.create_dataset(dataset2_cfg)
    

    cfg1['model']['datasets_cfgs'] = {dataset1_cfg['name']: dataset1.get_class_names()} 
    cfg2['model']['datasets_cfgs'] = {dataset2_cfg['name']: dataset2.get_class_names()} 
    
    model1 = model_factory.create_model(cfg1['model'])
    model1.freeze_all_heads()
    model2 = model_factory.create_model(cfg2['model'])
    model2.freeze_all_heads()
    
    pt_weights1 = copy.deepcopy(model1.state_dict())
    pt_weights1 = OrderedDict((k, v) for k, v in pt_weights1.items() if "classifier_heads" not in k)
    
    pt_weights2 = copy.deepcopy(model1.state_dict())
    pt_weights2 = OrderedDict((k, v) for k, v in pt_weights2.items() if "classifier_heads" not in k)
    
    dataset1_cfg['train_transforms'] = model1.get_val_transforms()
    dataset1_cfg['val_transforms'] = model1.get_val_transforms()
    dataset1, num_classes1 = dataset_factory.create_dataset(dataset1_cfg)
    dataset2_cfg['train_transforms'] = model1.get_val_transforms()
    dataset2_cfg['val_transforms'] = model1.get_val_transforms()
    dataset2, num_classes2 = dataset_factory.create_dataset(dataset2_cfg)
    
    dataset1.reset_train_dl(shuffle=False)
    dataset2.reset_train_dl(shuffle=False)
    
    
    strategy1 = cfg1['strategy']
    strategy2 = cfg2['strategy']
    noise_tv1 = strategy1['noise']['finetuning'][0]
    noise_tv1['set'] = 'Heldout'
    # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
    if noise_tv1['noise_type'] == 'asymmetric':
        dataset1.inject_noise(**noise_tv1)
        hs_clean, hs_noisy = dataset1.get_clean_noisy_subsets(set='Heldout')
        dataset1.switch_labels_to_clean(hs_noisy)
        
        dataset1.set_heldoutset(hs_noisy, shuffle=False)
    
        dataset1_clean = copy.deepcopy(dataset1)
    
        dataset1.inject_noise(**strategy1['noise']['pretraining'])
        ho_set = dataset1.get_heldoutset()
        dataset1.switch_labels_to_noisy(ho_set)
        dataset1.set_heldoutset(ho_set)
    else:
        dataset1_clean = copy.deepcopy(dataset1)
        dataset1.inject_noise(**strategy1['noise']['pretraining'])
        dataset1.inject_noise(**noise_tv1)


    # Load weights while removing classifier weights from the state dict
    mix_weights1 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs1_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    gold_weights1 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs1_dir.joinpath(f"clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_ho_clean_weights1 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs1_dir.joinpath(f"finetune_clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    mix_weights2 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs2_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    gold_weights2 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs2_dir.joinpath(f"clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_ho_clean_weights2 = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs2_dir.joinpath(f"finetune_clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    
    noise_weights1 = OrderedDict()
    for noise_tv in strategy1['noise']['finetuning']:
        ft_expr_dir = outputs1_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        noise_weights1[f"Seed {noise_tv['seed']}"] = n_weights
        
    noise_weights2 = OrderedDict()
    for noise_tv in strategy2['noise']['finetuning']:
        ft_expr_dir = outputs2_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        noise_weights2[f"Seed {noise_tv['seed']}"] = n_weights
    
            
    task_vectors1 = OrderedDict()
    for task_name, finetuend_weights in noise_weights1.items():
        task_vectors1[task_name] = TaskVector(mix_weights1, finetuend_weights)
        
    if len(task_vectors1) == 1:
        only_tv = task_vectors1.popitem(last=False)[1]
        task_vectors1['Average'] = only_tv
    else:
        task_vectors1['Average'] = TaskVector.mean(task_vectors1)
        
    
    task_vectors1['Clean'] = TaskVector(mix_weights1, ft_ho_clean_weights1)
    task_vectors1['Mix'] = TaskVector(pt_weights1, mix_weights1)
    
    task_vectors1['Random Vector'] = task_vectors1['Average'].generate_random_vector_with_same_layer_norms(seed=training_seed)
    
    
    task_vectors2 = OrderedDict()
    for task_name, finetuend_weights in noise_weights2.items():
        task_vectors2[task_name] = TaskVector(mix_weights2, finetuend_weights)
        
    if len(task_vectors2) == 1:
        only_tv = task_vectors2.popitem(last=False)[1]
        task_vectors2['Average'] = only_tv
    else:
        task_vectors2['Average'] = TaskVector.mean(task_vectors2)
        
    
    task_vectors2['Clean'] = TaskVector(mix_weights2, ft_ho_clean_weights2)
    task_vectors2['Mix'] = TaskVector(pt_weights2, mix_weights2)
    
    task_vectors2['Random Vector'] = task_vectors2['Average'].generate_random_vector_with_same_layer_norms(seed=training_seed)



    model2.load_state_dict(pt_weights2, strict=False)
    pt_results2, _, _ = evaluate_model(model2, dataset2.get_test_dataloader(), gpu)
    print(pt_results2)
    
    model2.load_state_dict(pt_weights1, strict=False)
    pt_results1, _, _ = evaluate_model(model2, dataset2.get_test_dataloader(), gpu)
    print(pt_results1)
    
    model2.load_state_dict(mix_weights1, strict=False)
    mix_results1, _, _ = evaluate_model(model2, dataset2.get_test_dataloader(), gpu)
    print(mix_results1)
    
    task_vectors1['Average'].apply_to(model2, scaling_coef=1.0, strict=False)
    proxy_results1, _, _ = evaluate_model(model2, dataset2.get_test_dataloader(), gpu)
    print(proxy_results1)
    


from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        nargs=2,                    # require exactly two
        metavar=("CONFIG1", "CONFIG2"),
        type=str,
        help="Two configurations use for model and dataset loading.",
    )
    
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg1_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config[0]}.yaml"
    cfg2_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config[1]}.yaml"

    if not cfg1_path.exists(): raise RuntimeError('The specified config1 file does not exist.')
    if not cfg2_path.exists(): raise RuntimeError('The specified config2 file does not exist.')
    with open(cfg1_path, 'r') as file:
        cfg1 = yaml.full_load(file)
    with open(cfg2_path, 'r') as file:
        cfg2 = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_noise_TA").absolute()
    results_dir = Path("results/single_experiment/clip_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    global_seed = cfg1['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])

        
    apply_tv(outputs_dir, results_dir, cfg1, cfg2, cfg1_name=cfg1_path.stem, cfg2_name=cfg2_path.stem)

if __name__ == "__main__":
    main()