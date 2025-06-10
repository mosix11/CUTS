import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, make_resnet18k, FCN
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt

from src.utils import nn_utils, misc_utils
import torch
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv
import yaml

import copy

from ray import train, tune
from ray.tune import TuneConfig, RunConfig, FailureConfig



def process_dataset(cfg, augmentations=None, phase:str='pretraining'):
    cfg['dataset']['batch_size'] = cfg['trainer'][phase]['batch_size']
    del cfg['trainer'][phase]['batch_size']
    dataset_name = cfg['dataset'].pop('name')
    cfg['dataset']['augmentations'] = augmentations if augmentations else []
    
    if dataset_name == 'mnist':
        pass
    elif dataset_name == 'cifar10':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg['dataset']
        )
    elif dataset_name == 'cifar100':
        pass
    elif dataset_name == 'mog':
        pass
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes


def process_model(cfg, num_classes):
    model_type = cfg['model'].pop('type')
    if cfg['model']['loss_fn'] == 'MSE':
        cfg['model']['loss_fn'] = torch.nn.MSELoss()
    elif cfg['model']['loss_fn'] == 'CE':
        cfg['model']['loss_fn'] = torch.nn.CrossEntropyLoss()
    else: raise ValueError(f"Invalid loss function {cfg['model']['loss_fn']}.")
    
    
    if cfg['model']['metrics']:
        metrics = {}
        for metric_name in cfg['model']['metrics']:
            if metric_name == 'ACC':
                metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            elif metric_name == 'F1':
                metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
            else: raise ValueError(f"Invalid metric {metric_name}.")
        cfg['model']['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)
    elif model_type == 'fcN':
        model = FCN(**cfg)
    elif model_type == 'cnn5':
        model = CNN5(**cfg['model'])
    elif model_type == 'resnet18k':
        model = make_resnet18k(**cfg)
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model

def unwrap_noise_configurations(config_dict):
    """
    Unwraps the lists inside 'strategy:noise' from a loaded YAML dictionary.

    Args:
        config_dict (dict): The dictionary loaded from the YAML file.

    Returns:
        tuple: A tuple containing two lists:
            - list_a (list): Each dictionary in this list represents a configuration
                             with a single 'pretraining' noise setting. The 'finetuning'
                             noise setting is removed.
            - list_b (list): Each dictionary in this list represents a combination
                             of single 'pretraining' and 'finetuning' noise settings,
                             where the 'noise_rate' of finetuning is greater than
                             that of pretraining.
    """
    list_a = []
    list_b = []

    # Safely access the noise configurations
    strategy_noise = config_dict.get('strategy', {}).get('noise', {})
    pretraining_configs = strategy_noise.get('pretraining', [])
    finetuning_configs = strategy_noise.get('finetuning', [])

    # Generate list_a (pretraining only configurations)
    for pretraining_item in pretraining_configs:
        new_config = copy.deepcopy(config_dict)
        new_config['strategy']['noise']['pretraining'] = pretraining_item
        # Remove finetuning if it exists
        if 'finetuning' in new_config['strategy']['noise']:
            del new_config['strategy']['noise']['finetuning']

        list_a.append(new_config)

    # Generate list_b (pretraining + finetuning combinations with noise_rate condition)
    for pretraining_item in pretraining_configs:
        for finetuning_item in finetuning_configs:
            if finetuning_item.get('noise_rate', 0.0) > pretraining_item.get('noise_rate', 0.0):
                new_config = copy.deepcopy(config_dict)
                new_config['strategy']['noise']['pretraining'] = pretraining_item
                new_config['strategy']['noise']['finetuning'] = finetuning_item
                list_b.append(new_config)

    return list_a, list_b

def apply_strategy(cfg, dataset, phase:str):
    strategy = cfg['strategy']
    # if strategy['finetuning_set'] == 'TrainingSet':
    #     pass
    # elif strategy['finetuning_set'] == 'HeldoutSet':
    #     pass
    # else: raise ValueError(f"Invalid strategy type {strategy['finetuning_set']}.")    
    dataset.inject_noise(**strategy['noise'][phase])
    if phase == 'finetuning' and strategy['finetuning_set'] == 'Heldout':
        dataset.replace_heldout_as_train_dl()
        
    return dataset



def pretrain_trainable(config, outputs_dir:Path):
    cfg = config['cfg']
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = process_dataset(cfg, augmentations, phase='pretraining')
    model = process_model(cfg, num_classes) 
    dataset = apply_strategy(cfg, dataset, phase='pretraining')
    
    
    experiment_name = f"pn={cfg['strategy']['noise']['pretraining']['noise_rate']}_pretrain"
    experiment_tags = experiment_name.split("_")

    experiment_dir = outputs_dir / Path(experiment_name)

    weights_dir = experiment_dir / Path("weights")
    weights_dir.mkdir(exist_ok=True, parents=True)

    plots_dir = experiment_dir / Path("plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        **cfg['trainer']['pretraining'],
        exp_name=experiment_name,
        exp_tags=experiment_tags,
    )
    
    results = trainer.fit(model, dataset, resume=False)
    torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))
    
    class_names = [f"Class {i}" for i in range(num_classes)]
    confmat = trainer.confmat("Test", num_classes=num_classes)
    misc_utils.plot_confusion_matrix(
        cm=confmat,
        class_names=class_names,
        filepath=str(plots_dir / Path("confmat.png")),
        show=False
    )
    

def finetune_trainable(config, outputs_dir:Path):
    cfg = config['cfg']
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = process_dataset(cfg, augmentations, phase='finetuning')
    model = process_model(cfg, num_classes) 
    dataset = apply_strategy(cfg, dataset, phase='finetuning')
    
    
    pt_expr_name = f"pn={cfg['strategy']['noise']['pretraining']['noise_rate']}_pretrain"
    pt_expr_dir = outputs_dir / Path(pt_expr_name)
    
    base_model_ckp_path = pt_expr_dir / Path('weights/model_weights.pth')
    checkpoint = torch.load(base_model_ckp_path)
    model.load_state_dict(checkpoint)

    ft_expr_name = f"pn={cfg['strategy']['noise']['pretraining']['noise_rate']}|fn={cfg['strategy']['noise']['finetuning']['noise_rate']}_finetune"
    ft_expr_dir = outputs_dir / Path(ft_expr_name)

    weights_dir = ft_expr_dir / Path("weights")
    weights_dir.mkdir(exist_ok=True, parents=True)

    plots_dir = ft_expr_dir / Path("plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        **cfg['trainer']['finetuning'],
        exp_name=ft_expr_name,
        exp_tags=ft_expr_name.split('_'),
    )
    
    results = trainer.fit(model, dataset, resume=False)
    torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))
    
    class_names = [f"Class {i}" for i in range(num_classes)]
    confmat = trainer.confmat("Test", num_classes=num_classes)
    misc_utils.plot_confusion_matrix(
        cm=confmat,
        class_names=class_names,
        filepath=str(plots_dir / Path("confmat.png")),
        show=False
    ) 

def do_analysis(outputs_dir: Path, cfg: dict, cfg_name:str):
    
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    outputs_dir = outputs_dir / Path(cfg_name)
    
    pretrain_cfgs, finetune_cfgs = unwrap_noise_configurations(cfg)
    # print(len(pretrain_cfgs), len(finetune_cfgs))
    # print(pretrain_cfgs[0])
    # print('\n\n\n')
    # print(pretrain_cfgs[1])
    # print('\n\n\n')
    # print(finetune_cfgs[0])
    gpu_per_experiment:float = 0.2
    cpu_per_experiment:float = 4
    

    configs = {
        "cfg": tune.grid_search(pretrain_cfgs)
    }
    resources_per_expr = {"cpu": cpu_per_experiment, "gpu": gpu_per_experiment}
    trainable_with_gpu_resources = tune.with_resources(
        tune.with_parameters(pretrain_trainable, outputs_dir=outputs_dir),
        resources=resources_per_expr
    )
    
    ray_strg_dir = outputs_dir / Path('ray/prterain')
    ray_strg_dir.mkdir(exist_ok=True, parents=True)
    tuner = tune.Tuner(
        trainable_with_gpu_resources, # Your trainable function wrapped with resources
        param_space=configs,    # The hyperparameters to explore
        tune_config=TuneConfig(
            scheduler=None
        ),
        run_config=RunConfig(
            name=None, # Name for this experiment run
            storage_path=str(ray_strg_dir.absolute()), # Default location for results
            failure_config=FailureConfig(
                max_failures=-1 # -1: Continue running other trials if one fails
                                # 0 (Default): Stop entire run if one trial fails
            )
        )
    )
    pretrain_results = tuner.fit()
    
    
    
    configs = {
        "cfg": tune.grid_search(finetune_cfgs)
    }
    resources_per_expr = {"cpu": cpu_per_experiment, "gpu": gpu_per_experiment}
    trainable_with_gpu_resources = tune.with_resources(
        tune.with_parameters(finetune_trainable, outputs_dir=outputs_dir),
        resources=resources_per_expr
    )
    
    ray_strg_dir = outputs_dir / Path('ray/finetune')
    ray_strg_dir.mkdir(exist_ok=True, parents=True)
    tuner = tune.Tuner(
        trainable_with_gpu_resources, # Your trainable function wrapped with resources
        param_space=configs,    # The hyperparameters to explore
        tune_config=TuneConfig(
            scheduler=None
        ),
        run_config=RunConfig(
            name=None, # Name for this experiment run
            storage_path=str(ray_strg_dir.absolute()), # Default location for results
            failure_config=FailureConfig(
                max_failures=-1 # -1: Continue running other trials if one fails
                                # 0 (Default): Stop entire run if one trial fails
            )
        )
    )
    finetune_results = tuner.fit()





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
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/grid_analysis').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/grid_analysis").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    do_analysis(outputs_dir, cfg, cfg_name=cfg_path.stem)
