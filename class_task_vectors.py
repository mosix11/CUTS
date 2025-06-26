import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory
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
import pickle
import copy



def finetune_model(outputs_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    
    dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations, phase='finetuning')
    
    pt_model = model_factory.create_model(cfg['model'], num_classes)
    
    base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}_pretrain") / Path('weights/model_weights.pth')
    pt_weights = torch.load(base_model_ckp_path)
    pt_model.load_state_dict(pt_weights)
    

    
    strategy = cfg['strategy']
    for class_idx in range(num_classes):
        
        cfg_copy = copy.deepcopy(cfg)
        
        experiment_name = f"{cfg_name}_finetune/class{class_idx}"
        experiment_tags = experiment_name.split("_")

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        ft_model = model_factory.create_model(cfg_copy['ft_model'], num_classes)
        ft_model.load_backbone_weights(state_dict=pt_weights)
        
        ft_dataset = copy.deepcopy(dataset)
        
        if 'noise' in strategy:
            pass
        
        if strategy['finetuning_set'] == 'ClassTVBinaryHead':
            ft_dataset.binarize_set('Train', target_class=class_idx)
            ft_dataset.binarize_set('Test', target_class=class_idx)
        
        # trainset = ft_dataset.get_trainset()
        # num = 0
        # for item in trainset:
        #     y = item[1]
        #     if y:
        #         num+=1
        # print('Class', class_idx, num)

        
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            **cfg_copy['trainer']['finetuning'],
            exp_name=experiment_name,
            exp_tags=experiment_tags,
        )
        
        results = trainer.fit(ft_model, ft_dataset, resume=False)
        print(results)

        torch.save(ft_model.state_dict(), weights_dir / Path("model_weights.pth"))

        # class_names = [f"Class {i}" for i in range(num_classes)]
        # confmat = trainer.confmat("Test")
        # misc_utils.plot_confusion_matrix(
        #     cm=confmat,
        #     class_names=class_names,
        #     filepath=str(plots_dir / Path("confmat.png")),
        # )


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

    finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
