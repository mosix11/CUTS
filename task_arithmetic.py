import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, resnet18k, FCN
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


class TaskVector:
    def __init__(self, pretrained_state_dict=None, finetuned_state_dict=None, vector=None):
        """
        Initialize a TaskVector using either:
        - pretrained_state_dict and finetuned_state_dict, OR
        - an existing vector (dictionary of parameter deltas).
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None, \
                "Provide either vector or both state_dicts."
            self.vector = {}
            with torch.no_grad():
                for key in pretrained_state_dict:
                    if key not in finetuned_state_dict:
                        print(f"Warning: key {key} missing in finetuned_state_dict.")
                        continue
                    if pretrained_state_dict[key].dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        continue  # Skip non-float entries
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors."""
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning: key {key} not found in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        return self if other in (None, 0) else self.__add__(other)

    def __neg__(self):
        """Negate the task vector."""
        with torch.no_grad():
            neg_vector = {key: -val for key, val in self.vector.items()}
        return TaskVector(vector=neg_vector)

    def apply_to(self, model, scaling_coef=1.0, strict=False):
        """
        Applies the task vector to the weights of an existing model.
        
        Parameters:
        - model: a PyTorch model with pre-loaded weights (state_dict already applied)
        - scaling_coef: float multiplier for the task vector (default: 1.0)
        - strict: if True, will raise errors for missing/unmatched keys
        """
        with torch.no_grad():
            updated_state_dict = model.state_dict()
            for key in self.vector:
                if key not in updated_state_dict:
                    if strict:
                        raise KeyError(f"Key {key} not found in model state_dict.")
                    else:
                        print(f"Warning: key {key} not found in model. Skipping.")
                        continue
                updated_state_dict[key] = updated_state_dict[key] + scaling_coef * self.vector[key]
            model.load_state_dict(updated_state_dict, strict=False)
        return model

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', filename='confusion_matrix_tv.png'):
    """
    Plots the confusion matrix and saves it to a file.

    Args:
        cm (np.ndarray): The confusion matrix (2D numpy array).
        class_names (list, optional): A list of class names to display on the axes.
                                        If None, will use 0, 1, 2...
        title (str): The title of the plot.
        filename (str): The path and filename to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Close the plot to free memory


def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def train_fc_cifar10(outputs_dir: Path):
    max_epochs = 600
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
    log_comet = True
    
    use_amp = True
    
    # augmentations = [
    #     transformsv2.RandomCrop(32, padding=4),
    #     transformsv2.RandomHorizontalFlip()
    # ]
    
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
        # augmentations=augmentations,
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
    precision_metric = torchmetrics.Precision(task='multiclass', num_classes=10)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes=10)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10)
    metrics = {
        'ACC': acc_metric,
        'Precision': precision_metric,
        'Recall': recall_metric,
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
    
    base_model_ckp = torch.load(base_model_dir / 'checkpoint/final_ckp.pth', map_location=cpu)["model_state"]
    finetune_model_ckp = torch.load(finetune_model_dir / 'checkpoint/final_ckp.pth', map_location=cpu)["model_state"]
    
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
    plot_confusion_matrix(cm=cm, class_names=class_names)
    
    print(f"Loss of the negated model is {loss_met.avg}")
    print(f"Metrics of the negated model is {metric_results}")

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
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        help="Whether to run experiments in parallel.",
        action="store_true"
    )
    args = parser.parse_args()
    
    dotenv.load_dotenv('.env')
    
    
    outputs_dir = Path("outputs/").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc" and args.dataset == "mnist":
        if args.parallel:
            # train_fc1_mnist_parallel(outputs_dir)
            pass
        else:
            # train_fc1_mnist(outputs_dir)
            pass
    elif args.model == "fc" and args.dataset == "cifar10":
        train_fc_cifar10(outputs_dir)
        pass
    elif args.model == "fc" and args.dataset == "mog":
        if args.parallel:
            # train_fc1_mog_parallel(outputs_dir)
            pass
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        if args.parallel:
            # train_cnn5_cifar10_parallel(outputs_dir)
            pass
        else:
            # train_cnn5_cifar10(outputs_dir)
            pass
        
    elif args.model == 'resnet18k' and args.dataset == 'cifar10':
        pass