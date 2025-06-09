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

from PIL import Image

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




def train_cnn5_cifar10(outputs_dir: Path):
    max_epochs = 200
    batch_size = 1024
    class_subset = []
    remap_labels = False
    balance_classes = True
    label_noise = 0.0
    normalize_imgs = True
    
    training_seed = 11
    dataset_seed = 11
    
    dataset = CIFAR10(
        batch_size=batch_size,
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
    
    
    model_base = CNN5(
        num_channels=128,
        num_classes=10,
        # weight_init=weight_init_method,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    model_finetune = CNN5(
        num_channels=128,
        num_classes=10,
        # weight_init=weight_init_method,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    
    base_model_dir = outputs_dir / Path("cnn5|k128_cifar10|ln0.4|aug|full_seeds11+11_adam|lr0.0001|b1024|AMP_HalfTraining")
    finetune_model_dir = outputs_dir / Path("cnn5|k128_cifar10|ln0.4|aug|full_seeds11+11_adam|lr0.0001|b1024|AMP_HalfTraining_FineTune")
    
    base_model_ckp = torch.load(base_model_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_model_ckp = torch.load(finetune_model_dir / 'weights/model_weights.pth', map_location=cpu)
    
    model_base.load_state_dict(base_model_ckp)
    model_finetune.load_state_dict(finetune_model_ckp)
    
    task_vector = TaskVector(
        pretrained_state_dict=base_model_ckp,
        finetuned_state_dict=finetune_model_ckp
    )
    
    task_vector.apply_to(model_base, scaling_coef=-0.35)
    
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
    misc_utils.plot_confusion_matrix(cm=cm, class_names=class_names, filepath='confusion_matrix_tv.png', show=False)
    
    plot_multiple_confusion_matrices(
        filepath1=base_model_dir / Path('plots/confmat.png'),
        filepath2=finetune_model_dir / Path('plots/confmat.png'),
        filepath3='confusion_matrix_tv.png',
        title1='Pretrained',
        title2='Finetuned',
        title3='TV',
        save_filepath='confusion_matrices_combined.png'
    )
    
    print(f"Loss of the negated model is {loss_met.avg}")
    print(f"Metrics of the negated model is {metric_results}")


# {'final': {'Train/Loss': 0.01166448979973793, 'Train/ACC': 0.9990500211715698, 'Train/F1': 0.9990500211715698, 'Test/Loss': 1.1495949382781983, 'Test/ACC': 0.6852999925613403, 'Test/F1': 0.6852999925613403}, 'best': {'Train/Loss': 1.2972837326049804, 'Train/LR': 0.0001, 'Train/ACC': 0.6348000168800354, 'Train/F1': 0.6348000168800354, 'Val/Loss': 1.669271377182007, 'Val/ACC': 0.5419999957084656, 'Val/F1': 0.5419999957084656, 'epoch': 38}}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for training.",
        type=str,
        choices=["fc", "cnn5", "resnet18k"],
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="The dataset used for trainig the model.",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "mog"],
        required=True,
    )

    args = parser.parse_args()
    
    dotenv.load_dotenv('.env')
    
    
    outputs_dir = Path("outputs/").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc" and args.dataset == "mnist":
        pass
    elif args.model == "fc" and args.dataset == "cifar10":
        train_fc_cifar10(outputs_dir)
        pass
    elif args.model == "fc" and args.dataset == "mog":
        pass
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        train_cnn5_cifar10(outputs_dir)
            
        
    elif args.model == 'resnet18k' and args.dataset == 'cifar10':
        pass