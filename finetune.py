import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic, data_utils
from src.models import FC1, CNN5, make_resnet18k, FCN
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
import seaborn as sns
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



def process_dataset(cfg, augmentations=None):
    cfg['dataset']['batch_size'] = cfg['trainer']['finetuning']['batch_size']
    del cfg['trainer']['finetuning']['batch_size']
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


def apply_strategy(cfg, dataset):
    strategy = cfg['strategy']
    # if strategy['finetuning_set'] == 'TrainingSet':
    #     pass
    # elif strategy['finetuning_set'] == 'HeldoutSet':
    #     pass
    # else: raise ValueError(f"Invalid strategy type {strategy['finetuning_set']}.")
    
    dataset.inject_noise(**strategy['noise']['finetuning'])
    if strategy['finetuning_set'] == 'Heldout':
        dataset.replace_heldout_as_train_dl()
    return dataset


def finetune_model(outputs_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = process_dataset(cfg, augmentations)
    
    model = process_model(cfg, num_classes)
    
    dataset = apply_strategy(cfg, dataset)

    base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}_pretrain") / Path('weights/model_weights.pth')
    checkpoint = torch.load(base_model_ckp_path)
    model.load_state_dict(checkpoint)
    
    experiment_name = f"{cfg_name}_finetune"
    experiment_tags = experiment_name.split("_")

    experiment_dir = outputs_dir / Path(experiment_name)

    weights_dir = experiment_dir / Path("weights")
    weights_dir.mkdir(exist_ok=True, parents=True)

    plots_dir = experiment_dir / Path("plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        **cfg['trainer']['finetuning'],
        exp_name=experiment_name,
        exp_tags=experiment_tags,
    )
    
    results = trainer.fit(model, dataset, resume=False)
    print(results)

    torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

    class_names = [f"Class {i}" for i in range(num_classes)]
    confmat = trainer.confmat("Test", num_classes=num_classes)
    misc_utils.plot_confusion_matrix(
        cm=confmat,
        class_names=class_names,
        filepath=str(plots_dir / Path("confmat.png")),
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
    
    cfg_path = Path('configs').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
