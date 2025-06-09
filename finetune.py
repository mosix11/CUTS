import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic, data_utils
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
import yaml



def train_fc_cifar10(outputs_dir: Path):
    max_epochs = 500
    # subsample_size = (30,1000)
    batch_size = 512
    img_size = (16,16)
    class_subset = []
    remap_labels = False
    balance_classes = True
    label_noise = 0.0
    
    # heldout_conf = (0.7, True)
    heldout_conf = {
        2: (0.9, False),
        5: (0.9, False)
    }
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
        heldout_conf=heldout_conf,
        # augmentations=augmentations,
        normalize_imgs=normalize_imgs,
        valset_ratio=0.0,
        
        num_workers=8,
        seed=dataset_seed,
    )
    
    dataset.train_loader = dataset.heldout_loader
    
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-7,
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
            metric=metrics,
        )
    
    # experiment_name = "fc1|h4096|p1093642_cifar10|ln0.0|noaug|full_seeds11+11_adam|lr0.0001|b1024|AMP"
    experiment_name = "fc1|h4096|p1093642_cifar10|ln0.0|noaug|full_seeds11+11_adam|lr0.0001|b1024|AMPFullTraining"
    
    base_model_ckp_path = outputs_dir/ Path(experiment_name) / Path('weights/model_weights.pth')
    checkpoint = torch.load(base_model_ckp_path)
    model.load_state_dict(checkpoint)
        
    
    experiment_name += '_FineTune'
    experiment_tags = experiment_name.split('_')
    
        
    experiment_dir = outputs_dir / Path(experiment_name)
        
    weights_dir = experiment_dir / Path('weights')
    weights_dir.mkdir(exist_ok=True, parents=True)
    
    plots_dir = experiment_dir / Path('plots')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    experiment_name += '_FineTune'
    
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
        pass
    elif model_type == 'fcN':
        pass
    elif model_type == 'cnn5':
        model = CNN5(
            **cfg['model']
        )
    elif model_type == 'resnet18k':
        pass
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model


def apply_strategy(cfg, dataset):
    strategy = cfg['strategy']
    # if strategy['finetuning_set'] == 'TrainingSet':
    #     pass
    # elif strategy['finetuning_set'] == 'HeldoutSet':
    #     pass
    # else: raise ValueError(f"Invalid strategy type {strategy['finetuning_set']}.")
    
    dataset.inject_noise(**strategy['noise']['finetuning'])
    if strategy['finetuning_set'] == 'Heldout':
        dataset.replace_heldout_as_train_dl()
    return dataset


def finetune_model(outputs_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]
    dataset, num_classes = process_dataset(cfg, augmentations)
    
    model = process_model(cfg, num_classes)
    
    dataset = apply_strategy(cfg, dataset)

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
    confmat = trainer.confmat("Test", num_classes=num_classes)
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
    
    cfg_path = Path('configs').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    finetune_model(outputs_dir, cfg, cfg_name=cfg_path.stem)
