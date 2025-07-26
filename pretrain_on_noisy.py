import comet_ml
from src.datasets import dataset_factory, data_utils
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
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
import json
from tqdm import tqdm
from collections import OrderedDict




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
            
            loss, preds = model.validation_step(input_batch, target_batch, use_amp=True, return_preds=True)
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            predictions = torch.argmax(preds, dim=-1)
            all_preds.extend(predictions.cpu())
            all_targets.extend(target_batch.cpu())
            
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return metric_results, torch.tensor(all_preds), torch.tensor(all_targets) 


def search_optimal_coefficient(base_model, task_vector, search_range, dataset, num_classes, device):
    """
    Performs a search to find the optimal task vector scaling coefficient.

    Args:
        base_model (torch.nn.Module): The pre-trained model. A deepcopy is made for each evaluation.
        task_vector (TaskVector): The task vector object.
        dataset: The dataset object to get the test dataloader from.
        search_range (list or tuple): A list/tuple [min_val, max_val] for the search.
        device (torch.device): The device to run evaluation on.
        num_classes (int): The number of classes for the confusion matrix.

    Returns:
        tuple: (best_coefficient, best_performance_metrics, confusion_matrix_tensor)
    """
    test_dataloader = dataset.get_test_dataloader()
    
    best_coef = 0.0
    best_acc = -1.0
    best_results = {}
    
    print("--- Starting Coarse Search ---")
    coarse_search_grid = np.arange(search_range[0], search_range[1] + 0.1, 0.1)
    
    for scale_coef in tqdm(coarse_search_grid, desc="Coarse Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results
    
    # print(f"\nCoarse search best coefficient: {best_coef:.2f} with Accuracy: {best_acc:.4f}")

    print("\n--- Starting Fine Search ---")
    fine_search_start = max(search_range[0], best_coef - 0.1)
    fine_search_end = min(search_range[1], best_coef + 0.1)
    fine_search_grid = np.linspace(fine_search_start, fine_search_end, num=21)

    for scale_coef in tqdm(fine_search_grid, desc="Fine Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results

    # print(f"\nRecalculating metrics and confusion matrix for best coefficient: {best_coef:.2f}")
    final_model = copy.deepcopy(base_model)
    task_vector.apply_to(final_model, scaling_coef=best_coef)
    final_model.to(device)

    best_results, all_preds, all_targets = evaluate_model(final_model, test_dataloader, device)
    
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    best_cm_tensor = confmat_metric(all_preds, all_targets)

    return best_coef, best_results, best_cm_tensor


def eval_model_on_clean_noise_splits(model, cfg, dataset, device):
    dataset_cpy = copy.deepcopy(dataset)
    strategy = cfg['strategy']
    dataset_cpy.inject_noise(**strategy['noise']['pretraining'])
    clean_set, noisy_set = dataset_cpy.get_clean_noisy_subsets(set='Train')
    
    dataset_cpy.set_trainset(clean_set, shuffle=False)
    clean_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    noisy_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)
    
    dummy_instance = noisy_set
    while not isinstance(dummy_instance, data_utils.NoisyClassificationDataset):
        dummy_instance = dummy_instance.dataset
    dummy_instance.switch_to_clean_lables()
    
    dataset_cpy.set_trainset(noisy_set, shuffle=False)
    healing_metric, _, _ = evaluate_model(model, dataloader=dataset_cpy.get_train_dataloader(), device=device)

    
    return {
        'clean_set': clean_metric,
        'noisy_set': noisy_metric,
        'healing_noise': healing_metric,
    }
    
    
    
def generate_latex_table_from_results(results_dict, output_path):
    def format_acc(val):
        return f"{val * 100:.2f}"

    def format_loss(val):
        return f"{val:.3f}"

    def get_nested(d, *keys):
        for key in keys:
            d = d.get(key, {})
        return d

    def latex_delta(current, baseline, positive_good=True):
        delta = (current - baseline) * 100
        if abs(delta) < 1e-2:
            return ""
        color = "green" if (delta > 0 and positive_good) or (delta < 0 and not positive_good) else "red"
        sign = "+" if delta > 0 else "-"
        return f" \\textcolor{{{color}}}{{({sign}{abs(delta):.2f})}}"

    # Baselines from "pretrain"
    pretrain = results_dict["pretrain"]
    base_utility_acc = get_nested(pretrain, "test_results", "ACC")
    base_forgetting_acc = get_nested(pretrain, "train_results", "noisy_set", "ACC")
    base_healing_acc = get_nested(pretrain, "train_results", "healing_noise", "ACC")
    base_clean_acc = get_nested(pretrain, "train_results", "clean_set", "ACC")

    # Begin LaTeX table
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Utility} & \multicolumn{2}{c}{Forgetting Rate} & \multicolumn{2}{c}{Healing Rate} & \multicolumn{2}{c}{Performance on $\mathcal{D}_{clean}$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"Model & ACC & Loss & ACC & Loss & ACC & Loss & ACC & Loss \\",
        r"\midrule"
    ]

    for name, data in results_dict.items():

        test_acc = get_nested(data, "test_results", "ACC")
        test_loss = get_nested(data, "test_results", "Loss")

        noisy_acc = get_nested(data, "train_results", "noisy_set", "ACC")
        noisy_loss = get_nested(data, "train_results", "noisy_set", "Loss")

        healing_acc = get_nested(data, "train_results", "healing_noise", "ACC")
        healing_loss = get_nested(data, "train_results", "healing_noise", "Loss")

        clean_acc = get_nested(data, "train_results", "clean_set", "ACC")
        clean_loss = get_nested(data, "train_results", "clean_set", "Loss")

        # Format values
        acc_util_str = format_acc(test_acc) + latex_delta(test_acc, base_utility_acc, positive_good=True)
        acc_forgot_str = format_acc(noisy_acc) + latex_delta(noisy_acc, base_forgetting_acc, positive_good=False)
        acc_heal_str = format_acc(healing_acc) + latex_delta(healing_acc, base_healing_acc, positive_good=True)
        acc_clean_str = format_acc(clean_acc) + latex_delta(clean_acc, base_clean_acc, positive_good=True)

        loss_util_str = format_loss(test_loss)
        loss_forgot_str = format_loss(noisy_loss)
        loss_heal_str = format_loss(healing_loss)
        loss_clean_str = format_loss(clean_loss)

        # Escape underscores in names
        safe_name = r"\texttt{" + name.replace("_", r"\_") + r"}"

        # Add row
        lines.append(
            f"{safe_name} & {acc_util_str} & {loss_util_str} & "
            f"{acc_forgot_str} & {loss_forgot_str} & "
            f"{acc_heal_str} & {loss_heal_str} & "
            f"{acc_clean_str} & {loss_clean_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Performance metrics from different model states. Accuracy values are shown as percentages. Differences from the baseline (\texttt{pretrain}) are marked in \textcolor{green}{green} for improvements and \textcolor{red}{red} for regressions.}",
        r"\label{tab:noise_unlearning_results}",
        r"\end{table}"
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table successfully written to {output_path}")
    
def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        results[tv_name][-1.0] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=-1.0)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
        base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][-1.0]['test_results'] = base_test_results
        results[tv_name][-1.0]['train_results'] = base_train_split_results
        
        base_model = copy.deepcopy(model)

        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=tv,
            search_range=(-3.0, 0.0),
            dataset=dataset,
            num_classes=num_classes,
            device=device
        )
        
        results[tv_name][best_coef] = OrderedDict()
        results[tv_name][best_coef]['test_results'] = best_results
        
        tv.apply_to(base_model, scaling_coef=best_coef)
        
        after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][best_coef]['train_results'] = after_tv_metrics
        
        
    avg_tv = None
    count = 0  
    for tv_name, tv in taskvectors.items():
        if tv_name != "ft_gt_noise":
            count += 1
            if avg_tv is not None:  
                avg_tv += (tv - avg_tv) * (1/count)
            else:
                avg_tv = tv
                
    results['avg_noise'] = OrderedDict()
    base_model = copy.deepcopy(model)
    results['avg_noise'][-1.0] = OrderedDict()
    avg_tv.apply_to(base_model, scaling_coef=-1.0)
    base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
    base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
    results['avg_noise'][-1.0]['test_results'] = base_test_results
    results['avg_noise'][-1.0]['train_results'] = base_train_split_results
    
    base_model = copy.deepcopy(model)

    best_coef, best_results, best_cm = search_optimal_coefficient(
        base_model=base_model,
        task_vector=avg_tv,
        search_range=(-3.0, 0.0),
        dataset=dataset,
        num_classes=num_classes,
        device=device
    )
    
    results['avg_noise'][best_coef] = OrderedDict()
    results['avg_noise'][best_coef]['test_results'] = best_results
    avg_tv.apply_to(base_model, scaling_coef=best_coef)
    after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
    results['avg_noise'][best_coef]['train_results'] = after_tv_metrics
    
    
    random_vec = avg_tv.generate_random_vector_with_same_layer_norms(seed=11)
    results['rnd_vec'] = OrderedDict()
    
    base_model = copy.deepcopy(model)
    results['rnd_vec'][-1.0] = OrderedDict()
    random_vec.apply_to(base_model, scaling_coef=-1.0)
    base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
    base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
    results['rnd_vec'][-1.0]['test_results'] = base_test_results
    results['rnd_vec'][-1.0]['train_results'] = base_train_split_results
    
    base_model = copy.deepcopy(model)

    best_coef, best_results, best_cm = search_optimal_coefficient(
        base_model=base_model,
        task_vector=random_vec,
        search_range=(-3.0, 0.0),
        dataset=dataset,
        num_classes=num_classes,
        device=device
    )
    
    results['rnd_vec'][best_coef] = OrderedDict()
    results['rnd_vec'][best_coef]['test_results'] = best_results
    base_model = copy.deepcopy(model)
    random_vec.apply_to(base_model, scaling_coef=best_coef)
    after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
    results['rnd_vec'][best_coef]['train_results'] = after_tv_metrics
    
    return results

def pt_ft_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")

    if cfg['dataset']['name'] == 'cifar10':
        # augmentations = None
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'cifar100':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'mnist':
        # augmentations = None
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]

    
    if not outputs_dir.joinpath(f"{cfg_name}/gold/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/gold"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['pretraining'],
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
        

    if not outputs_dir.joinpath(f"{cfg_name}/pretrain/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        
        dataset.inject_noise(**strategy['noise']['pretraining'])
        
        experiment_name = f"{cfg_name}/pretrain"

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['pretraining'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        if cfg['strategy']['finetuning_set'] == 'LowLoss':
            trainer.setup_data_loaders(dataset)
            trainer.activate_low_loss_samples_buffer(
                consistency_window=5,
                consistency_threshold=0.8
            )

        results = trainer.fit(model, dataset, resume=False)
        
        # print(results)

        torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False,
        )


    if not outputs_dir.joinpath(f"{cfg_name}/finetune_gold/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
        checkpoint = torch.load(base_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/finetune_gold"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['pretraining'],
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
        
        
    if not outputs_dir.joinpath(f"{cfg_name}/finetune_gt_noise/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
        checkpoint = torch.load(base_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(noisy_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/finetune_gt_noise"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['finetuning'],
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

    
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        print(cfg['strategy'])
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth").exists():
            cfg_cpy = copy.deepcopy(cfg)
            dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
            
            model = model_factory.create_model(cfg_cpy['model'], num_classes)
            
            base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
            checkpoint = torch.load(base_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            strategy = cfg_cpy['strategy']
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout':
                dataset.inject_noise(**noise_tv)
                dataset.replace_heldout_as_train_dl()
            elif strategy['finetuning_set'] == 'CleanNoiseSplit':
                dataset.inject_noise(**strategy['noise']['pretraining'])
                clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
                if noise_tv['set'] == 'TrainClean':
                    dataset.set_trainset(clean_set, shuffle=True)
                    noise_tv['set'] = 'Train'
                    dataset.inject_noise(**noise_tv)
                
                elif noise_tv['set'] == 'TrainNoise':
                    dataset.set_trainset(noisy_set, shuffle=True)
                    
            elif strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{strategy['percentage']:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.inject_noise(**strategy['noise']['pretraining'])
                dataset.subset_set(set='Train', indices=all_easy_samples)
                
                dataset.inject_noise(**noise_tv)
                
            elif strategy['finetuning_set'] == 'HighLoss':
                pass
            
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['finetuning'],
                exp_name=experiment_name,
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
            
            
def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    base_expr_dir = outputs_dir / cfg_name
    gold_dir = base_expr_dir / 'gold'
    pretrain_dir = base_expr_dir / 'pretrain'
    ft_gold_dir = base_expr_dir / 'finetune_gold'
    ft_gt_noise_dir = base_expr_dir / 'finetune_gt_noise'
    finetune_dirs = OrderedDict()
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        ft_expr_dir = base_expr_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        finetune_dirs[f"{noise_tv['noise_rate']}_{noise_tv['seed']}"] = ft_expr_dir
        
    gold_weights = torch.load(gold_dir / 'weights/model_weights.pth', map_location=cpu)
    pretrain_weights = torch.load(pretrain_dir / 'weights/model_weights.pth', map_location=cpu)
    ft_gold_wieghts = torch.load(ft_gold_dir / 'weights/model_weights.pth', map_location=cpu)
    ft_gt_noise_weights = torch.load(ft_gt_noise_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_dirs.items():
        finetune_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
    
    
    
    ft_gold_tv = TaskVector(pretrain_weights, ft_gold_wieghts)
    ft_gt_noise_tv = TaskVector(pretrain_weights, ft_gt_noise_weights)
    finetune_tvs = OrderedDict()
    for ft_expr, ft_weight in finetune_weights.items():
        finetune_tvs[ft_expr] = TaskVector(pretrain_weights, ft_weight)
        
    ft_tvs_list = [ft_gold_tv, ft_gt_noise_tv]
    # ft_tvs_list = []
    ft_tvs_list.extend(list(finetune_tvs.values()))
    print(finetune_tvs.keys())
    # print(ft_gold_tv.layer_wise_cosine_similarity(ft_tvs_list[1]))

    
    # for key, weight_tensor in ft_gt_noise_tv.vector.items():
    #     if 'bn' in key:
    #         print(key, weight_tensor.sum())
    # exit()
    
    tv_names = ['Gold', 'Ground Truth Noise']
    # class_names = []
    tv_names.extend([f"{float(n_s.split('_')[0])*100:.0f}% Noise, {n_s.split('_')[1]} Seed" for n_s in list(finetune_tvs.keys())])
    
    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
    misc_utils.plot_confusion_matrix(
        title='Task Vector Similarity Matrix',
        cm=task_sim,
        class_names=tv_names,
        color_map='vlag',
        color_bar=True,
        vmin= -1.0,
        vmax= 1.0,
        x_label='Task Vectors',
        y_label='Task Vectors',
        tick_label_font_size=6,
        filepath=results_dir / 'task_similarities.png',
        show=False
    )
    
    
    
    # for i in range(len(ft_tvs_list)):
    #     if i == 0:
    #         print('passing ft gold from low rank approximation')
    #         continue
    #     else:
    #         ftsv = ft_tvs_list[i].compute_SVD_for_each_layer(k=0.1)
        
    #         ft_tvs_list[i].apply_SVD_to_TV(ftsv)
    
    
    # task_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = ft_tvs_list[i]
    #     task_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = ft_tvs_list[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         task_sim[i].append(cos_sim)
    # task_sim = np.array(task_sim)
    
    # misc_utils.plot_confusion_matrix(cm=task_sim, class_names=class_names, filepath=None, show=True)
    
    # for ft_name, ft_tv in finetune_tvs.items():
    #     best_coef, best_results, best_cm = search_optimal_coefficient(
    #         base_model=base_model,
    #         task_vector=ft_tv,
    #         search_range=(-1.5, 0.0),
    #         dataset=dataset,
    #         num_classes=num_classes,
    #         device=gpu
    #     )
    #     print(f"Best scaling coefficient for {ft_name} = {best_coef}")
    #     print(f"Metrics of the negated model is {best_results}")
            
    
    

        
    # TSV = TaskVector.TSV_extract_common_direction(finetune_tvs, k=0.3)
    # TSV.apply_to(base_model, scaling_coef=1.0)
        
    # for ft_name, ft_tv in finetune_tvs.items():
    #     best_coef, best_results, best_cm = search_optimal_coefficient(
    #         base_model=base_model,
    #         task_vector=ft_tv,
    #         search_range=(-1.5, 0.0),
    #         dataset=dataset,
    #         num_classes=num_classes,
    #         device=gpu
    #     )
    #     print(f"Best scaling coefficient for {ft_name} = {best_coef}")
    #     print(f"Metrics of the negated model is {best_results}")
    
    # test_tv = (ft_tvs_list[2] + ft_tvs_list[3]) * 0.5
    
    # ordered_dict = OrderedDict(zip(tv_names[1:], ft_tvs_list[1:]))
    
    # best_coef, best_results, best_cm = search_optimal_coefficient(
    #     base_model=base_model,
    #     # task_vector=test_tv,
    #     task_vector=ft_tvs_list[4],
    #     search_range=(-3.0, 0.0),
    #     dataset=dataset,
    #     num_classes=num_classes,
    #     device=gpu
    # )
    
    # print(f"Best scaling coefficient for TV = {best_coef}")
    # print(f"Metrics of the negated model is {best_results}")
    
    # before_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print('Performance before TV:', before_tv_metrics)
    # base_model.to(cpu)
    # # test_tv.apply_to(base_model, scaling_coef=best_coef)
    # ft_tvs_list[4].apply_to(base_model, scaling_coef=best_coef)
    
    # after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print('Performance after TV:', after_tv_metrics)
    
    base_model.load_state_dict(pretrain_weights)
    pt_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    pt_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(gold_weights)
    gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    gold_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(ft_gold_wieghts)
    ft_gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    ft_gold_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(pretrain_weights)
    base_model.to(cpu)
    
    results_dict = OrderedDict()
    
    results_dict['pretrain'] = {'test_results': pt_test_results, 'train_results': pt_train_results}
    results_dict['gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
    results_dict['ft_gold'] = {'test_results': ft_gold_test_results, 'train_results': ft_gold_train_results}
    results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names[1:], ft_tvs_list[1:])), results_dict, cfg, dataset, num_classes, gpu)
    
    
    print(results_dict)
    
    
    with open(results_dir / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    # generate_latex_table_from_results(results_dict, results_dir / 'results_tex.txt')
    
    # otrh_tvs, shrd_tvs = TaskVector.decompose_task_vectors_SVD(ft_tvs_list)
    
    # orth_task_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = otrh_tvs[i]
    #     orth_task_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = otrh_tvs[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         orth_task_sim[i].append(cos_sim)
    # orth_task_sim = np.array(orth_task_sim)
    
    # shrd_tvs_sim = []
    # for i in range(len(ft_tvs_list)):
    #     anchor_tv = shrd_tvs[i]
    #     shrd_tvs_sim.append([])
    #     for j in range(len(ft_tvs_list)):
    #         other_tv = shrd_tvs[j]
    #         cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
    #         shrd_tvs_sim[i].append(cos_sim)
    # shrd_tvs_sim = np.array(shrd_tvs_sim)
    
    
    
    # misc_utils.plot_confusion_matrix(cm=orth_task_sim, class_names=class_names, filepath=None, show=True)
    # misc_utils.plot_confusion_matrix(cm=shrd_tvs_sim, class_names=class_names, filepath=None, show=True)
    
    


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
        "-t",
        "--tv",
        help="Apply task vectors to an already trained and finetuned experiment.",
        action="store_true",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/pretrain_on_noisy') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/pretrain_on_noisy").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/pretrain_on_noisy").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    else:
        pt_ft_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)