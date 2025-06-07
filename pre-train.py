import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, resnet18k, FCN
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
from src.utils import nn_utils
import torch
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv


def train_fc1_cifar10(outputs_dir: Path):
    max_epochs = 2000
    subsample_size = (10,1000)
    batch_size = 256
    img_size = (16,16)
    class_subset = [2, 5]
    remap_labels = True
    balance_classes = True
    label_noise = 0.4
    
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
        subsample_size=subsample_size,
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
    
    train_dl = dataset.get_train_dataloader()
    
    labels = set()
    num_tot = 0
    num_noisy = 0
    for batch in train_dl:
        x, y, is_noisy = batch
        # print(x.shape, y.shape, is_noisy.shape)
        print(y)
        print(is_noisy)
        num_tot += x.shape[0]
        num_noisy += torch.sum(is_noisy).item()
        labels.update(y.tolist())
    print(labels)
    print(num_tot)
    print(num_noisy)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for training.",
        type=str,
        choices=["fc1", "cnn5", "resnet18k"],
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
    
    
    outputs_dir = Path("outputs/modelwise").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        if args.parallel:
            # train_fc1_mnist_parallel(outputs_dir)
            pass
        else:
            # train_fc1_mnist(outputs_dir)
            pass
    elif args.model == "fc1" and args.dataset == "cifar10":
        train_fc1_cifar10(outputs_dir)
        pass
    elif args.model == "fc1" and args.dataset == "mog":
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