import comet_ml
from src.datasets import dataset_factory, CIFAR10, CIFAR100
from src.models import FC1, CNN5, make_resnet18k, FCN, model_factory
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
import yaml
 
 
dotenv.load_dotenv(".env") 

COMET_API_KEY = os.getenv("COMET_API_KEY")

training_seed = 11
dataset_seed = 11
noise_seed = 8


outputs_dir = Path('outputs/single_experiment')


 
augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip(),
    ]


num_classes = 100
dataset = CIFAR100(
    batch_size=1024,
    img_size=[32,32],
    augmentations=augmentations,
    normalize_imgs=True,
    flatten=False,
    valset_ratio=0.0,
    return_index=True,
    num_workers=8,
    seed=dataset_seed
)

dataset.inject_noise(
    set='Train',
    noise_rate=0.2,
    noise_type='symmetric',
    seed=noise_seed
)
clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
dataset.set_trainset(clean_set, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'ACC': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    'F1': torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
}


expr_name = f'gold_cifar{num_classes}_cleanset_0.2noise_seed{noise_seed}'

model = make_resnet18k(
    init_channels=64,
    num_classes=num_classes,
    loss_fn=loss_fn,
    metrics=metrics
)

optim_cfg = {
    'type': 'adamw',
    'lr': 1e-4,
    'betas': (0.9, 0.999)
}

trainer = TrainerEp(
    outputs_dir=outputs_dir,
    max_epochs=600,
    optimizer_cfg=optim_cfg,
    save_best_model=False,
    log_comet=True,
    comet_api_key=COMET_API_KEY,
    comet_project_name='gold_standards',
    exp_name=expr_name,
    seed=training_seed,
)

results = trainer.fit(model, dataset, resume=False)

print(results)
experiment_dir = outputs_dir / Path(expr_name)

weights_dir = experiment_dir / Path("weights")
weights_dir.mkdir(exist_ok=True, parents=True)

plots_dir = experiment_dir / Path("plots")
plots_dir.mkdir(exist_ok=True, parents=True)

torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

class_names = [f"Class {i}" for i in range(num_classes)]
confmat = trainer.confmat("Test")
misc_utils.plot_confusion_matrix(
    cm=confmat,
    class_names=class_names,
    filepath=str(plots_dir / Path("confmat.png")),
)