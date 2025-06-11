import comet_ml
from src.datasets import dataset_factory
from src.models import FC1, CNN5, make_resnet18k, FCN, model_factory
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
 
    


def apply_strategy(cfg, dataset):
    strategy = cfg['strategy']
    # if strategy['finetuning_set'] == 'TrainingSet':
    #     pass
    # elif strategy['finetuning_set'] == 'HeldoutSet':
    #     pass
    # else: raise ValueError(f"Invalid strategy type {strategy['finetuning_set']}.")
    
    dataset.inject_noise(**strategy['noise']['pretraining'])
    
    return dataset


def pretrain_model(outputs_dir: Path, cfg: dict, cfg_name:str):

    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)
    
    model = model_factory.create_model(cfg, num_classes)
    
    dataset = apply_strategy(cfg, dataset)
    
    experiment_name = f"{cfg_name}_pretrain"
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
    
    cfg_path = Path('configs/single_experiment').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    pretrain_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
