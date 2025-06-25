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


def apply_strategy(cfg, dataset, pretrain_expr_dir:Path, phase:str="finetuning"):
    strategy = cfg['strategy']
    # if strategy['finetuning_set'] == 'TrainingSet':
    #     pass
    # elif strategy['finetuning_set'] == 'HeldoutSet':
    #     pass
    # else: raise ValueError(f"Invalid strategy type {strategy['finetuning_set']}.")    
    
    if phase == 'finetuning' and strategy['finetuning_set'] == 'Heldout':
        dataset.inject_noise(**strategy['noise'][phase])
        dataset.replace_heldout_as_train_dl()
    elif phase == 'finetuning' and strategy['finetuning_set'] == 'CleanNoiseSplit':
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        if strategy['noise']['finetuning']['set'] == 'TrainClean':
            dataset.set_trainset(clean_set, shuffle=True)
            strategy['noise']['finetuning']['set'] = 'Train'
            dataset.inject_noise(**strategy['noise']['finetuning'])
        
        elif strategy['noise']['finetuning']['set'] == 'TrainNoise':
            dataset.set_trainset(noisy_set, shuffle=True)
            
    elif phase == 'finetuning' and strategy['finetuning_set'] == 'LowLoss':
        # low_loss_idxs_path = pretrain_expr_dir / f'log/low_loss_indices_{strategy['noise']['pretraining']['noise_rate']}.pkl'
        low_loss_idxs_path = pretrain_expr_dir / f'log/low_loss_indices_{strategy['percentage']}.pkl'
        with open(low_loss_idxs_path, 'rb') as mfile:
            low_loss_indices = pickle.load(mfile)
        all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
        
        dataset.inject_noise(**strategy['noise']['pretraining'])
        dataset.subset_set(set='Train', indices=all_easy_samples)
        
        dataset.inject_noise(**strategy['noise']['finetuning'])
        
        # num = 0
        # trainset = dataset.get_trainset()
        # for item in trainset:
        #     if item[2] == True:
        #         num+=1
        # print(len(trainset))
        # print(num)
        
        # exit()
        
        # num_clean = 0
        # num_noisy = 0
        # for item in clean_set:
        #     x, y, is_noisy, idx = item
        #     if not is_noisy: num_clean+=1
        #     else: print('Booooogh clean')
        # for item in noisy_set:
        #     x, y, is_noisy, idx = item
        #     if is_noisy: num_noisy+=1
        #     else: print('Booooogh noisy')
        # print('Clean size', len(clean_set), 'Clean num', num_clean)
        # print('Noisy size', len(noisy_set), 'Noisy num', num_noisy)
        
        
    return dataset


def finetune_model(outputs_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations, phase='finetuning')
    
    model = model_factory.create_model(cfg, num_classes)
    
    dataset = apply_strategy(cfg, dataset, outputs_dir / Path(f"{cfg_name}_pretrain"))
    
    
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
    confmat = trainer.confmat("Test")
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

    finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
