import comet_ml
from src.datasets import dataset_factory, data_utils
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
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

from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient








def train_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['comet_api_key'] = os.getenv("COMET_API_KEY")

    augmentations = None
    if cfg['dataset']['name'] == 'cifar10':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'cifar100':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    elif cfg['dataset']['name'] == 'fashion_mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]

    base_dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)
    base_model = model_factory.create_model(cfg['model'], num_classes)
    strategy = cfg['strategy']
    base_dataset.inject_noise(**strategy['noise'])
    
    # For validation
    base_dataset.set_valset(base_dataset.get_testset())
    
    if not outputs_dir.joinpath(f"{cfg_name}/gold/weights/model_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/gold"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer'],
            exp_name=experiment_name,
            exp_tags=None,
        )

        results = trainer.fit(model, dataset, resume=False)

        torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False,
        )
        
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
            **cfg['trainer'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        

        # trainer.setup_data_loaders(dataset)
        # trainer.activate_low_loss_samples_buffer(
        #     consistency_window=5,
        #     consistency_threshold=0.8
        # )

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
    
    cfg_path = Path('configs/single_experiment/best_vs_gold') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/best_vs_gold").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/best_vs_gold").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    train_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)