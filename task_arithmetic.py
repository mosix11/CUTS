import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, resnet18k, FCN, TaskVector
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
import numpy as np
import random
from torchmetrics import ConfusionMatrix
from tqdm import tqdm
import yaml
from PIL import Image
import copy
import json

def plot_multiple_confusion_matrices(
    filepath1,
    filepath2,
    filepath3,
    title1=None,
    title2=None,
    title3=None,
    main_title='Combined Confusion Matrices',
    save_filepath=None,
    show=True
):
    """
    Combines and displays three confusion matrix plots vertically in a single figure.

    Args:
        filepath1 (str): Path to the first confusion matrix image.
        filepath2 (str): Path to the second confusion matrix image.
        filepath3 (str): Path to the third confusion matrix image.
        title1 (str, optional): Title for the first confusion matrix.
        title2 (str, optional): Title for the second confusion matrix.
        title3 (str, optional): Title for the third confusion matrix.
        main_title (str): Overall title for the combined figure. Defaults to 'Combined Confusion Matrices'.
        save_filepath (str, optional): Path to save the combined figure. If None, the figure is not saved.
        show (bool): If True, display the combined figure. Defaults to True.
    """
    try:
        img1 = Image.open(filepath1)
        img2 = Image.open(filepath2)
        img3 = Image.open(filepath3)
    except FileNotFoundError as e:
        print(f"Error: One or more confusion matrix image files not found. {e}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 18)) # 3 rows, 1 column, adjust figsize as needed

    # Plot the first confusion matrix
    axes[0].imshow(img1)
    axes[0].axis('off')  # Turn off axis labels and ticks for the image
    if title1:
        axes[0].text(-0.1, 0.5, title1, transform=axes[0].transAxes,
                     fontsize=12, va='center', ha='right', rotation=90) # Vertical title on the left

    # Plot the second confusion matrix
    axes[1].imshow(img2)
    axes[1].axis('off')
    if title2:
        axes[1].text(-0.1, 0.5, title2, transform=axes[1].transAxes,
                     fontsize=12, va='center', ha='right', rotation=90)

    # Plot the third confusion matrix
    axes[2].imshow(img3)
    axes[2].axis('off')
    if title3:
        axes[2].text(-0.1, 0.5, title3, transform=axes[2].transAxes,
                     fontsize=12, va='center', ha='right', rotation=90)

    fig.suptitle(main_title, fontsize=16, y=1.02) # Overall title at the top
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.98]) # Adjust layout to make space for main title and side titles

    if save_filepath:
        plt.savefig(save_filepath, bbox_inches='tight')

    if show:
        plt.show()
    
    plt.close() # Close the figure to free memory



def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def train_fc_cifar10(outputs_dir: Path):
    # subsample_size = (30,1000)
    batch_size = 1024
    img_size = (16,16)
    class_subset = []
    remap_labels = False
    balance_classes = True
    label_noise = 0.0
    
    # heldout_conf = (0.7, True)
    
    grayscale = True
    flatten = True
    normalize_imgs = False
    
    training_seed = 11
    dataset_seed = 11
    
    use_amp = True

    dataset = CIFAR10(
        batch_size=batch_size,
        # subsample_size=subsample_size,
        img_size=img_size,
        grayscale=grayscale,
        flatten=flatten,
        class_subset=class_subset,
        remap_labels=remap_labels,
        balance_classes=balance_classes,
        label_noise=label_noise,
        normalize_imgs=normalize_imgs,
        valset_ratio=0.0,
        
        num_workers=8,
        seed=dataset_seed,
    )
    
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
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10)
    metrics = {
        'ACC': acc_metric,
        'F1': f1_metric
    }
    # weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
    
    model_base = FC1(
            input_dim=256,
            hidden_dim=4096,
            output_dim=10,
            # weight_init=weight_init_method,
            loss_fn=loss_fn,
            metrics=metrics,
        )
    
    model_finetune = FC1(
            input_dim=256,
            hidden_dim=4096,
            output_dim=10,
            # weight_init=weight_init_method,
            loss_fn=loss_fn,
            metrics=metrics,
        )
    
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    base_model_dir = outputs_dir / Path("fc1|h4096|p1093642_cifar10|ln0.0|noaug|full_seeds11+11_adam|lr0.0001|b1024|AMPFullTraining")
    finetune_model_dir = outputs_dir / Path("fc1|h4096|p1093642_cifar10|ln0.0|noaug|full_seeds11+11_adam|lr0.0001|b1024|AMPFullTraining_FineTune")
    
    base_model_ckp = torch.load(base_model_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_model_ckp = torch.load(finetune_model_dir / 'weights/model_weights.pth', map_location=cpu)
    
    model_base.load_state_dict(base_model_ckp)
    model_finetune.load_state_dict(finetune_model_ckp)
    
    task_vector = TaskVector(
        pretrained_state_dict=base_model_ckp,
        finetuned_state_dict=finetune_model_ckp
    )
    
    
    task_vector.apply_to(model_base, scaling_coef=-0.5)
    
    model_base.to(gpu)
    model_base.eval()
    
    test_dataloader = dataset.get_test_dataloader()
    
    loss_met = misc_utils.AverageMeter()
    model_base.reset_metrics()
    all_preds = []
    all_targets = []
    
    for i, batch in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
            ):
        input_batch, target_batch, is_noisy = prepare_batch(batch, gpu)
        loss = model_base.validation_step(input_batch, target_batch, use_amp=True)
        loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
        
        model_output = model_base.predict(input_batch)
        predictions = torch.argmax(model_output, dim=-1) 
        
        all_preds.extend(predictions.detach().cpu())
        all_targets.extend(target_batch.detach().cpu())
        
    metric_results = model_base.compute_metrics()
    model_base.reset_metrics()
    confmat = ConfusionMatrix(task="multiclass", num_classes=10)
    cm = confmat(torch.tensor(all_preds), torch.tensor(all_targets))
    
    class_names = [f'Class {i}' for i in range(10)]
    misc_utils.plot_confusion_matrix(cm=cm, class_names=class_names)
    
    print(f"Loss of the negated model is {loss_met.avg}")
    print(f"Metrics of the negated model is {metric_results}")





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
        pass
    elif model_type == 'cnn5':
        model = CNN5(**cfg['model'])
    elif model_type == 'resnet18k':
        pass
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model


def apply_tv(scale_coef:float, outputs_dir: Path, cfg: dict, cfg_name:str):
    
    dataset, num_classes = process_dataset(cfg)
    
    model_base = process_model(cfg, num_classes)
    model_ft = copy.deepcopy(model_base)
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    pretrain_dir = outputs_dir/ Path(f"{cfg_name}_pretrain")
    finetune_dir = outputs_dir/ Path(f"{cfg_name}_finetune")
    
    base_model_ckp_path = pretrain_dir / Path('weights/model_weights.pth')
    base_model_stat_dict = torch.load(base_model_ckp_path, map_location=cpu)
    model_base.load_state_dict(base_model_stat_dict)
    
    with open(pretrain_dir / Path('log/results.json'), 'r') as file:
        base_model_results = json.load(file)
    
    ft_model_ckp_path = finetune_dir / Path('weights/model_weights.pth')
    ft_model_state_dict = torch.load(ft_model_ckp_path, map_location=cpu)
    model_ft.load_state_dict(ft_model_state_dict)
    
    with open(finetune_dir / Path('log/results.json'), 'r') as file:
        ft_model_results = json.load(file)
    
    
    task_vector = TaskVector(
        pretrained_state_dict=base_model_stat_dict,
        finetuned_state_dict=ft_model_state_dict
    )
    
    task_vector.apply_to(model_base, scaling_coef=scale_coef)
    
    model_base.to(gpu)
    model_base.eval()
    
    test_dataloader = dataset.get_test_dataloader()
    
    loss_met = misc_utils.AverageMeter()
    model_base.reset_metrics()
    all_preds = []
    all_targets = []
    
    for i, batch in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
            ):
        batch = prepare_batch(batch, gpu)
        if len(batch) == 3:
                input_batch, target_batch, is_noisy = batch
        else:
            input_batch, target_batch = batch
        loss = model_base.validation_step(input_batch, target_batch, use_amp=True)
        loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
        
        model_output = model_base.predict(input_batch)
        predictions = torch.argmax(model_output, dim=-1) 
        
        all_preds.extend(predictions.detach().cpu())
        all_targets.extend(target_batch.detach().cpu())
        
    metric_results = model_base.compute_metrics()
    model_base.reset_metrics()
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=10)
    cm = confmat(torch.tensor(all_preds), torch.tensor(all_targets))
    
    class_names = [f'Class {i}' for i in range(10)]
    misc_utils.plot_confusion_matrix(cm=cm, class_names=class_names, filepath='confusion_matrix_tv.png', show=False)
    
    plot_multiple_confusion_matrices(
        filepath1=pretrain_dir / Path('plots/confmat.png'),
        filepath2=finetune_dir / Path('plots/confmat.png'),
        filepath3='confusion_matrix_tv.png',
        title1='Pretrained',
        title2='Finetuned',
        title3='TV',
        save_filepath=f"{cfg_name}_confmat_combined.png"
    )
    
    print(f"Loss of the negated model is {loss_met.avg}")
    print(f"Metrics of the negated model is {metric_results}")
    
    print('pretrained results : \n', base_model_results)
    
    

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
        "-s",
        "--scale",
        help="Scale coefficient for the task vector.",
        type=float,
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    apply_tv(args.scale, outputs_dir, cfg, cfg_name=cfg_path.stem)
