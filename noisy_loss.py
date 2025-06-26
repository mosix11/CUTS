import comet_ml
from src.datasets import dataset_factory, CIFAR10, CIFAR100
from src.models import FC1, CNN5, make_resnet18k, FCN, model_factory, CompoundLoss
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
import random
import numpy as np
 


dotenv.load_dotenv(".env") 

COMET_API_KEY = os.getenv("COMET_API_KEY")



def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch

def unpack_batch(batch, default_value=None):
        """
        Unpacks a list/tuple into four variables, assigning default_value
        to any variables that don't have corresponding items.
        """
        # x = batch[0] if len(batch) > 0 else default_value
        # y = batch[1] if len(batch) > 1 else default_value
        # is_noisy = batch[2] if len(batch) > 2 else default_value
        # idx = batch[3] if len(batch) > 3 else default_value
        x = batch[0]
        y = batch[1]
        is_noisy = batch[2] if len(batch) == 4 else torch.zeros_like(y)
        idx = batch[3] if len(batch) == 4 else batch[2]
        return x, y, is_noisy, idx

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
    all_preds_class = []
    all_preds_noisy = []
    all_targets_class = []
    all_targets_noisy = []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            input_batch, target_batch, is_noisy, idxs = unpack_batch(batch)
            
            loss = model.validation_step(input_batch, (target_batch, is_noisy), use_amp=True)
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            preds_class, preds_noise = model.predict(input_batch)
            preds_class = torch.argmax(preds_class, dim=-1) 
            preds_noise = torch.argmax(preds_noise, dim=-1)
            
            all_preds_class.extend(preds_class.cpu())
            all_preds_noisy.extend(preds_noise.cpu())
            all_targets_class.extend(target_batch.cpu())
            all_targets_noisy.extend(is_noisy.cpu())
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return metric_results, torch.tensor(all_preds_class), torch.tensor(all_preds_noisy), torch.tensor(all_targets_class), torch.tensor(all_targets_noisy)

training_seed = 11
dataset_seed = 11
pt_noise_seed = 8



outputs_dir = Path('outputs/single_experiment')
expr_name = 'noisy_loss_cifar10_0.4noise'

experiment_dir = outputs_dir / Path(expr_name)

weights_dir = experiment_dir / Path("weights")
weights_dir.mkdir(exist_ok=True, parents=True)

plots_dir = experiment_dir / Path("plots")
plots_dir.mkdir(exist_ok=True, parents=True)

pretrained_weights = torch.load(weights_dir / Path("pretrain_model_weights.pth"))

 
augmentations = [
    transformsv2.RandomCrop(32, padding=4),
    transformsv2.RandomHorizontalFlip(),
]


num_classes = 10
dataset_pt = CIFAR10(
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

dataset_pt.inject_noise(
    set='Train',
    noise_rate=0.4,
    noise_type='symmetric',
    seed=pt_noise_seed
)


loss_fn = torch.nn.CrossEntropyLoss(reduction=None)
metrics = {
    'ACC': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    'F1': torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
}

model = CNN5(
    num_channels=128,
    num_classes=num_classes,
    gray_scale=False,
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
    max_epochs=50,
    optimizer_cfg=optim_cfg,
    save_best_model=False,
    log_comet=True,
    comet_api_key=COMET_API_KEY,
    comet_project_name='noisy_loss_exeriments',
    exp_name=expr_name,
    seed=training_seed,
)

percentage = 0.4
trainer.setup_data_loaders(dataset_pt)
trainer.activate_low_loss_samples_buffer(
    percentage=percentage,
    consistency_window=5,
    consistency_threshold=0.8
)


trainer.fit(model, dataset)

for ft_noise_seed in range(10, 20):
    dataset_ft = CIFAR10(
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

    dataset_ft.inject_noise(
        set='Train',
        noise_rate=0.6,
        noise_type='symmetric',
        seed=ft_noise_seed
    )


    loss_fn = CompoundLoss()
    metrics = {
        'CE-ACC': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        'CE-F1': torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        'BN-ACC': torchmetrics.Accuracy(task="binary", num_classes=num_classes),
        'BN-F1': torchmetrics.F1Score(task="binary", num_classes=num_classes)
    }

    model = CNN5_BH(
        num_channels=128,
        num_classes=num_classes,
        gray_scale=False,
        loss_fn=loss_fn,
        metrics=metrics
    )

# model.load_pretrained_weights_from_old_cnn5(pretrained_weights)

# optim_cfg = {
#     'type': 'adamw',
#     'lr': 5e-5,
#     'betas': (0.9, 0.999)
# }

# trainer = TrainerEp(
#     outputs_dir=outputs_dir,
#     max_epochs=400,
#     optimizer_cfg=optim_cfg,
#     save_best_model=False,
#     log_comet=True,
#     comet_api_key=COMET_API_KEY,
#     comet_project_name='noisy_loss_exeriments',
#     exp_name=expr_name,
#     seed=training_seed,
# )

# results = trainer.fit(model, dataset_ft, resume=False)

# print(results)


# torch.save(model.state_dict(), weights_dir / Path("finetuned_model_weights.pth"))

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

finetuned_weights = torch.load(weights_dir / Path("finetuned_model_weights.pth"))

model.load_state_dict(finetuned_weights)

metrics, _, _, _, _ = evaluate_model(model, dataset_pt.get_test_dataloader(), device=gpu)
print(metrics)


# class_names = [f"Class {i}" for i in range(num_classes)]
# confmat = trainer.confmat("Test")
# misc_utils.plot_confusion_matrix(
#     cm=confmat,
#     class_names=class_names,
#     filepath=str(plots_dir / Path("confmat.png")),
# )