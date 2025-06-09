import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, resnet18k, FCN
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
    # heldout_conf = {
    #     0: (0.2, False),
    #     1: (0.2, False),
    #     2: (0.8, False),
    #     3: (0.2, False),
    #     4: (0.2, False),
    #     5: (0.8, False),
    #     6: (0.2, False),
    #     7: (0.2, False),
    #     8: (0.2, False),
    #     9: (0.2, False),
    # }
    heldout_conf = {
        2: (0.7, False),
        5: (0.7, False)
    }
    
    
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
        # heldout_conf=heldout_conf,
        # augmentations=augmentations,
        normalize_imgs=normalize_imgs,
        valset_ratio=0.0,
        
        num_workers=8,
        seed=dataset_seed,
    )
    
    
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10)
    metrics = {
        'ACC': acc_metric,
        'F1': f1_metric
    }
    
    # weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
    
    
    model = FC1(
        input_dim=256,
        hidden_dim=4096,
        output_dim=10,
        # weight_init=weight_init_method,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seeds{training_seed}+{dataset_seed}"
    experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}"
    if use_amp: 
        experiment_name += "|AMP" 
    else: 
        experiment_name += "|noAMP"
    if lr_schedule_cfg: 
        experiment_name += f"|{lr_schedule_cfg['type']}"
    experiment_name += '_FullTraining'
    experiment_tags = experiment_name.split('_')
    
    experiment_dir = outputs_dir / Path(experiment_name)
        
    weights_dir = experiment_dir / Path('weights')
    weights_dir.mkdir(exist_ok=True, parents=True)
    
    plots_dir = experiment_dir / Path('plots')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=False,
        validation_freq=1,
        save_best_model=True,
        run_on_gpu=True,
        use_amp=use_amp,
        batch_prog=False,
        log_comet=log_comet,
        comet_api_key=os.getenv('COMET_API_KEY'),
        comet_project_name='task-vectors-cifar10',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=training_seed
    )
    results = trainer.fit(model, dataset, resume=False)
    print(results)
    torch.save(model.state_dict(), weights_dir / Path('model_weights.pth'))
    
    class_names = [f'Class {i}' for i in range(10)]
    confmat = trainer.confmat('Test', num_classes=10)
    misc_utils.plot_confusion_matrix(cm=confmat, class_names=class_names, filepath= str(plots_dir / Path('confmat.png')))
    
    
    
    # train_dl = dataset.get_train_dataloader()
    
    # labels = set()
    # num_tot = 0
    # num_noisy = 0
    # for batch in train_dl:
    #     x, y, is_noisy = batch
    #     # print(x.shape, y.shape, is_noisy.shape)
    #     # print(y)
    #     # print(is_noisy)
    #     num_tot += x.shape[0]
    #     num_noisy += torch.sum(is_noisy).item()
    #     labels.update(y.tolist())
    # print(labels)
    # print(num_tot)
    # print(num_noisy)
    
    # print('\n\n\n')
    # held_dl = dataset.get_heldout_dataloader()
    
    # labels = set()
    # num_tot = 0
    # num_noisy = 0
    # for batch in held_dl:
    #     x, y, is_noisy = batch
    #     # print(x.shape, y.shape, is_noisy.shape)
    #     # print(y)
    #     # print(is_noisy)
    #     num_tot += x.shape[0]
    #     num_noisy += torch.sum(is_noisy).item()
    #     labels.update(y.tolist())
    # print(labels)
    # print(num_tot)
    # print(num_noisy)


def train_cnn5_cifar10(outputs_dir: Path):
    max_epochs = 600
    # subsample_size = (30,1000)
    batch_size = 1024

    class_subset = []
    remap_labels = False
    balance_classes = True
    label_noise = 0.4
    valset_ratio = 0.1
    normalize_imgs = True
    
    # heldout_conf = (0.7, True)
    # heldout_conf = {
    #     0: (0.2, False),
    #     1: (0.2, False),
    #     2: (0.8, False),
    #     3: (0.2, False),
    #     4: (0.2, False),
    #     5: (0.8, False),
    #     6: (0.2, False),
    #     7: (0.2, False),
    #     8: (0.2, False),
    #     9: (0.2, False),
    # }
    # heldout_conf = {
    #     2: (0.7, False),
    #     5: (0.7, False)
    # }
    heldout_conf = {
        0: (0.5, False),
        1: (0.5, False),
        2: (0.5, False),
        3: (0.5, False),
        4: (0.5, False),
        5: (0.5, False),
        6: (0.5, False),
        7: (0.5, False),
        8: (0.5, False),
        9: (0.5, False),
    }
    
    
    training_seed = 11
    dataset_seed = 11
    log_comet = True
    
    use_amp = True
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip()
    ]
    
    dataset = CIFAR10(
        batch_size=batch_size,
        class_subset=class_subset,
        remap_labels=remap_labels,
        balance_classes=balance_classes,
        label_noise=label_noise,
        heldout_conf=heldout_conf,
        augmentations=augmentations,
        normalize_imgs=normalize_imgs,
        valset_ratio=valset_ratio,
        num_workers=8,
        seed=dataset_seed,
    )
    
    
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10)
    metrics = {
        'ACC': acc_metric,
        'F1': f1_metric
    }
    
    
    model = CNN5(
        num_channels=128,
        num_classes=10,
        # weight_init=weight_init_method,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seeds{training_seed}+{dataset_seed}"
    experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}"
    if use_amp: 
        experiment_name += "|AMP" 
    else: 
        experiment_name += "|noAMP"
    if lr_schedule_cfg: 
        experiment_name += f"|{lr_schedule_cfg['type']}"
    experiment_name += '_HalfTraining'
    experiment_tags = experiment_name.split('_')
    
    experiment_dir = outputs_dir / Path(experiment_name)
    
    weights_dir = experiment_dir / Path('weights')
    weights_dir.mkdir(exist_ok=True, parents=True)
    
    plots_dir = experiment_dir / Path('plots')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=False,
        validation_freq=1,
        save_best_model=True,
        run_on_gpu=True,
        use_amp=use_amp,
        batch_prog=False,
        log_comet=log_comet,
        comet_api_key=os.getenv('COMET_API_KEY'),
        comet_project_name='task-vectors-cifar10',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=training_seed
    )
    results = trainer.fit(model, dataset, resume=False)
    print(results)
    
    torch.save(model.state_dict(), weights_dir / Path('model_weights.pth'))
    
    class_names = [f'Class {i}' for i in range(10)]
    confmat = trainer.confmat('Test', num_classes=10)
    misc_utils.plot_confusion_matrix(cm=confmat, class_names=class_names, filepath= str(plots_dir / Path('confmat.png')))
    
    

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