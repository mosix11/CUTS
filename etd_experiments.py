import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector
from src.trainers import ETDTrainer, StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
import torch
import torchmetrics
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


def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch

def evaluate_model(model, dataloader, device):
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: A tuple containing (all_predictions, all_targets, metrics_dict).
    """
    loss_met = misc_utils.AverageMeter()
    model.reset_metrics()
    all_preds = []
    all_targets = []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            # input_batch, target_batch = batch[:2]
            input_batch, target_batch, indices = batch[:3]
            
            # loss = model.validation_step(input_batch, target_batch, use_amp=True)
            loss = model.training_step(input_batch, target_batch, indices, use_amp=True)
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            model_output = model.predict(input_batch)
            predictions = torch.argmax(model_output, dim=-1) 
            
            all_preds.extend(predictions.cpu())
            all_targets.extend(target_batch.cpu())
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return metric_results, torch.tensor(all_preds), torch.tensor(all_targets) 




def eval_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    model = model_factory.create_model(cfg['model']['drop'], num_classes)
    
    dataset.inject_noise(**cfg['strategy']['noise'])
    clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    

    cleanset_dataloader = dataset._build_dataloader(clean_set)
    noisyset_dataloader = dataset._build_dataloader(noisy_set)

    
    experiment_name = f"{cfg_name}/"
    experiment_dir = outputs_dir / Path(experiment_name)
    weights_dir = experiment_dir / Path("weights")
    plots_dir = experiment_dir / Path("plots")

    trained_weights = torch.load(weights_dir / 'model_weights.pth', map_location=cpu)
    # print(trained_weights.keys())
    # trained_weights = torch.load(experiment_dir / 'checkpoint/final_ckp.pth', map_location=cpu)['model_state']
    # print(trained_weights.keys())
    
    model.load_state_dict(trained_weights)
    
    testset_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), device=gpu)
    cleanset_results, _, _ = evaluate_model(model, cleanset_dataloader, gpu)
    noisyset_results, _, _ = evaluate_model(model, noisyset_dataloader, gpu)
    
    print(testset_results)
    print(cleanset_results)
    print(noisyset_results)



    # clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    # print(len(clean_set), len(noisy_set))
    # cleanset_clean = 0
    # cleanset_noisy = 0
    # noisyset_clean = 0
    # noisyset_noisy = 0
    # for item in clean_set:
    #     x, y, idx, is_noisy = item
    #     if is_noisy:
    #         cleanset_noisy+=1
    #     else: cleanset_clean +=1
    
    # for item in noisy_set:
    #     x, y, idx, is_noisy = item
    #     if is_noisy:
    #         noisyset_noisy+=1
    #     else: noisyset_clean +=1
            
    # print(cleanset_clean, cleanset_noisy)
    # print(noisyset_noisy, noisyset_clean)
    
    # exit()
    

def train_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    
    dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)
    model = model_factory.create_model(cfg['model']['standard'], num_classes)
    
    dataset.inject_noise(**cfg['strategy']['noise'])

    
    experiment_name = f"{cfg_name}/"
    experiment_dir = outputs_dir / Path(experiment_name)

    weights_dir = experiment_dir / Path("weights")
    weights_dir.mkdir(exist_ok=True, parents=True)

    plots_dir = experiment_dir / Path("plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    
    trainer = ETDTrainer(
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    parser.add_argument(
        "-t",
        "--train",
        help="Train the model.",
        action="store_true",
    )
    
    
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    

    parser.add_argument(
        "-e",
        "--evaluate",
        help="Evaluate the model.",
        action="store_true",
    )
    
    
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/edt') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/edt").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/edt").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.train:
        train_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.evaluate:
        eval_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)