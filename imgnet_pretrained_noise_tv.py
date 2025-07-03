import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector
from src.trainers import TrainerEp, TrainerGS
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
            input_batch, target_batch = batch[:2]
            
            loss = model.validation_step(input_batch, target_batch, use_amp=True)
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




def finetune_model(outputs_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['ft_train']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['ft_noise']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    
    
    
    if not outputs_dir.joinpath(f"{cfg_name}/no_noise/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
    
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        experiment_name = f"{cfg_name}/no_noise"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['ft_train'],
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
    
        
        
    
    if not outputs_dir.joinpath(f"{cfg_name}/clean_set/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
    
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['task_vectors']['tv_train'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/clean_set"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['ft_train'],
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
        
        
    if not outputs_dir.joinpath(f"{cfg_name}/train_set/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
    
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['task_vectors']['tv_train'])
            
        experiment_name = f"{cfg_name}/train_set"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['ft_train'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        
        if strategy['finetuning_set'] == 'LowLoss':
            # percentage = strategy['percentage']
            trainer.setup_data_loaders(dataset)
            trainer.activate_low_loss_samples_buffer(
                # percentage=percentage,
                consistency_window=5,
                consistency_threshold=0.8
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
    
    
    for idx, noise_tv in enumerate(cfg['strategy']['task_vectors']['tv_noise']):
        
        if not outputs_dir.joinpath(f"{cfg_name}/noise_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth").exists():
            cfg_cpy = copy.deepcopy(cfg)
            dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
            
            model = model_factory.create_model(cfg_cpy['model'], num_classes)
            
            strategy = cfg_cpy['strategy']
            dataset.inject_noise(**strategy['task_vectors']['tv_train'])
            
            experiment_name = f"{cfg_name}/noise_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            
            low_loss_idxs_path = outputs_dir / f"{cfg_name}/train_set" / f"log/low_loss_indices_{strategy['percentage']:.2f}.pkl"
            with open(low_loss_idxs_path, 'rb') as mfile:
                low_loss_indices = pickle.load(mfile)
            all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
            
            dataset.subset_set(set='Train', indices=all_easy_samples)
            dataset.inject_noise(**noise_tv)
            
            trainer = TrainerEp(
                outputs_dir=outputs_dir,
                **cfg_cpy['trainer']['ft_noise'],
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
        




def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str, search_range = [-1.5, 0.0]):
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

    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    model_base = model_factory.create_model(cfg['model'], num_classes)
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    base_expr_dir = outputs_dir / cfg_name
    
    pretrain_weights = model_base.state_dict()
    no_noise_weights = torch.load(base_expr_dir / 'no_noise/weights/model_weights.pth', map_location=cpu)
    clean_set_weights = torch.load(base_expr_dir / 'clean_set/weights/model_weights.pth', map_location=cpu)
    train_set_weights = torch.load(base_expr_dir / 'train_set/weights/model_weights.pth', map_location=cpu)
    
    noisy_weights = {}
    for idx, noise_tv in enumerate(cfg['strategy']['task_vectors']['tv_noise']):
        noisy_weight = torch.load(base_expr_dir / f'noise_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth', map_location=cpu)
        noisy_weights[f'{noise_tv['noise_rate']}_{noise_tv['seed']}'] = noisy_weight
    
    no_noise_tv = TaskVector(pretrain_weights, no_noise_weights)
    clean_set_tv = TaskVector(pretrain_weights, clean_set_weights)
    train_set_tv = TaskVector(pretrain_weights, train_set_weights)
    noisy_tvs = {}
    for task, n_weights in noisy_weights.items():
        noisy_tvs[task] = TaskVector(pretrain_weights, n_weights)
        
    # pretrain_weights = torch.load(base_model_ckp_path, map_location=cpu)



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
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the pretrained model on the tasks.",
        action="store_true",
    )
    
    
    parser.add_argument(
        "-t",
        "--taskvector",
        help="Perform task arithmetics.",
        action="store_true",
    )
    
    
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/imgnet_pretrained_models') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/imgnet_pretrained_models").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/imgnet_pretrained_models").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.taskvector:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.finetune:
        finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)

