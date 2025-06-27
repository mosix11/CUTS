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




def pretrain_model(outputs_dir: Path, cfg: dict, cfg_name:str):

    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)

    
    model = model_factory.create_model(cfg['model'], num_classes)
    
    
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
    confmat = trainer.confmat("Test")
    misc_utils.plot_confusion_matrix(
        cm=confmat,
        class_names=class_names,
        filepath=str(plots_dir / Path("confmat.png")),
        show=False
    )
    

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
        
        ft_dataset = copy.deepcopy(dataset)
        heldout_set = ft_dataset.get_heldoutset()
        
        class_samples_indices = []
        for idx, sample in enumerate(heldout_set):
            if sample[1] == class_idx:
                class_samples_indices.append(idx)
                
        class_heldouts = Subset(heldout_set, class_samples_indices)
        
        trainset = ft_dataset.get_trainset()
        
        extended_trainset = ConcatDataset([trainset, class_heldouts])
        
        ft_dataset.set_trainset(extended_trainset, shuffle=True)
        
        # smpls = {k:0 for k in range(10)}
        
        # for sample in extended_trainset:
        #     y = sample[1]
        #     smpls[y] += 1
            
        # print(smpls)
        
        # continue
        
        cfg_copy = copy.deepcopy(cfg)
        
        experiment_name = f"{cfg_name}_finetune/class{class_idx}"
        experiment_tags = experiment_name.split("_")

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        ft_model = copy.deepcopy(pt_model)
        
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            **cfg_copy['trainer']['finetuning'],
            exp_name=experiment_name,
            exp_tags=experiment_tags,
        )
        
        results = trainer.fit(ft_model, ft_dataset, resume=False)
        print(results)

        torch.save(ft_model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False
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
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    pretrain_dir = outputs_dir/ Path(f"{cfg_name}_pretrain")
    finetune_dir = outputs_dir/ Path(f"{cfg_name}_finetune")
    
    dataset, num_classes = dataset_factory.create_dataset(cfg, phase='finetuning')
    
    pt_model = model_factory.create_model(cfg['model'], num_classes)
    
    base_model_ckp_path = pretrain_dir / Path('weights/model_weights.pth')
    pt_weights = torch.load(base_model_ckp_path, map_location=cpu)
    pt_model.load_state_dict(pt_weights)
    
    pt_weights = pt_model.state_dict()
    

    
    task_vectors = []
    
    for class_idx in range(num_classes):
        
        cfg_copy = copy.deepcopy(cfg)
        
        experiment_name = f"class{class_idx}"
        experiment_tags = experiment_name.split("_")
        

        experiment_dir = finetune_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        ft_state_dict = torch.load(weights_dir / 'model_weights.pth', map_location=cpu)
        
        tv = TaskVector(pt_weights, ft_state_dict)
        
        task_vectors.append(tv)
        
    
        
    for class_idx in range(num_classes):
        task_vectors[class_idx].apply_to(pt_model, scaling_coef=0.05, strict=True)
    
    # task_vectors[0].apply_to(pt_model, scaling_coef=1.0, strict=True)
    # task_vectors[1].apply_to(pt_model, scaling_coef=1.0, strict=True)
    
    metrics, all_preds, all_targets = evaluate_model(pt_model, dataset.get_test_dataloader(), device=gpu)
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    confmat = confmat_metric(all_preds, all_targets)
    
    class_names = [f'Class {i}' for i in range(10)]
    misc_utils.plot_confusion_matrix(cm=confmat, class_names=class_names, filepath=results_dir / Path('confusion_matrix_tv.png'), show=True)
    
    print(metrics)
        
    
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
        "-p",
        "--pretrain",
        help="Pretrain the model.",
        action="store_true",
    )
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the pretrained model.",
        action="store_true",
    )
    
    parser.add_argument(
        "-t",
        "--tv",
        help="Apply task vectors to pretrained and finetuned models.",
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
    results_dir = Path("results/single_experiment").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.pretrain:
        pretrain_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
    elif args.finetune:
        finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
