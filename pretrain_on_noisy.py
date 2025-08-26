import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector, weight_norm_analysis
from src.trainers import StandardTrainer, TrainerRLS, utils as trainer_utils
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import misc_utils
import torch

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
import re

from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, analyze_IC, get_confusion_matrix, estimate_T_from_confusion, symmetric_noise_detected, row_normalize

    
def generate_latex_table_from_results(results_dict, output_path):
    def format_acc(val):
        return f"{val * 100:.1f}"

    # def format_loss(val):
    #     return f"{val:.3f}"

    def latex_delta(current, baseline, positive_good=True):
        delta = (current - baseline) * 100
        if abs(delta) < 1e-2:
            return ""
        color = "green" if (delta > 0 and positive_good) or (delta < 0 and not positive_good) else "red"
        sign = "+" if delta > 0 else "-"
        return f" \\textcolor{{{color}}}{{({sign}{abs(delta):.1f})}}"


    # Begin LaTeX table
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.5}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Utility \%} & \multicolumn{2}{c}{Forgetting Rate \%} & \multicolumn{2}{c}{Healing Rate \%} & \multicolumn{2}{c}{Performance on $\mathcal{D}_{clean}$ \%} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"Model & $\alpha=-1.0$ & Opt & $\alpha=-1.0$ & Opt & $\alpha=-1.0$ & Opt & $\alpha=-1.0$ & Opt \\",
        r"\midrule"
    ]
    
    # Baselines from "pretrain"
    pretrain = results_dict.pop("Pretrain")
    base_utility_acc = pretrain["test_results"]["ACC"]
    base_forgetting_acc = pretrain["train_results"]["noisy_set"]["ACC"]
    base_healing_acc = pretrain["train_results"]["healing_noise"]["ACC"]
    base_clean_acc = pretrain["train_results"]["clean_set"]["ACC"]
    
    pretrain_name = r"$\mathcal{M}_{pt}$"
    lines.append(
            f"{pretrain_name} & {format_acc(base_utility_acc)} & - & "
            f"{format_acc(base_forgetting_acc)} & - & "
            f"{format_acc(base_healing_acc)} & - & "
            f"{format_acc(base_clean_acc)} & - \\\\"
        )
    
    
    gold = results_dict.pop("Gold")
    gold_utility_acc = gold["test_results"]["ACC"]
    gold_forgetting_acc = gold["train_results"]["noisy_set"]["ACC"]
    gold_healing_acc = gold["train_results"]["healing_noise"]["ACC"]
    gold_clean_acc = gold["train_results"]["clean_set"]["ACC"]
    
    clean_gold_name = r"$\mathcal{M}_{clean}$"
    lines.append(
            f"{clean_gold_name} & {format_acc(gold_utility_acc)} & - & "
            f"{format_acc(gold_forgetting_acc)} & - & "
            f"{format_acc(gold_healing_acc)} & - & "
            f"{format_acc(gold_clean_acc)} & - \\\\"
        )

    for name, data in results_dict.items():
        if name == 'Finetune Gold' or name == 'Ground Truth Noise':
            continue
        
        raw_test_acc = data["-1.0"]["test_results"]["ACC"]
        opt_test_acc = list(data.values())[1]["test_results"]["ACC"]

        raw_noisy_acc = data["-1.0"]["train_results"]["noisy_set"]["ACC"]
        opt_noisy_acc = list(data.values())[1]["train_results"]["noisy_set"]["ACC"]

        raw_healing_acc = data["-1.0"]["train_results"]["healing_noise"]["ACC"]
        opt_healing_acc = list(data.values())[1]["train_results"]["healing_noise"]["ACC"]

        raw_clean_acc = data["-1.0"]["train_results"]["clean_set"]["ACC"]
        opt_clean_acc = list(data.values())[1]["train_results"]["clean_set"]["ACC"]

        # Format values
        raw_acc_util_str = format_acc(raw_test_acc) + latex_delta(raw_test_acc, base_utility_acc, positive_good=True)
        opt_acc_util_str = format_acc(opt_test_acc) + latex_delta(opt_test_acc, base_utility_acc, positive_good=True)
        
        raw_acc_forgot_str = format_acc(raw_noisy_acc) + latex_delta(raw_noisy_acc, base_forgetting_acc, positive_good=False)
        opt_acc_forgot_str = format_acc(opt_noisy_acc) + latex_delta(opt_noisy_acc, base_forgetting_acc, positive_good=False)
        
        raw_acc_heal_str = format_acc(raw_healing_acc) + latex_delta(raw_healing_acc, base_healing_acc, positive_good=True)
        opt_acc_heal_str = format_acc(opt_healing_acc) + latex_delta(opt_healing_acc, base_healing_acc, positive_good=True)
        
        raw_acc_clean_str = format_acc(raw_clean_acc) + latex_delta(raw_clean_acc, base_clean_acc, positive_good=True)
        opt_acc_clean_str = format_acc(opt_clean_acc) + latex_delta(opt_clean_acc, base_clean_acc, positive_good=True)


        # Escape underscores in names
        single_noise_match = re.match(r'^\d+% Noise, (\d+) Seed$', name)
        pruned_avg_match = re.match(r'^Average TV Pruned ([0-9]*\.?[0-9]+)$', name)
        if single_noise_match:
            seed = int(single_noise_match.group(1))
            table_name = r"$\vec{T}^s_{" + str(seed) + r"}$"
        elif pruned_avg_match:
            prune_rate = float(pruned_avg_match.group(1))
            table_name = r"$\vec{T}^p_{" + str(prune_rate) + r"}$"
        elif name == 'Average TV':
            table_name = r"$\vec{T}_{avg}$"
        elif name == 'Random Vector':
            table_name = r"$\vec{T}_{rnd}$"
        else:
            raise ValueError('Vector name is not recognized', name)

        # Add row
        lines.append(
            f"{table_name} & {raw_acc_util_str} & {opt_acc_util_str} & "
            f"{raw_acc_forgot_str} & {opt_acc_forgot_str} & "
            f"{raw_acc_heal_str} & {opt_acc_heal_str} & "
            f"{raw_acc_clean_str} & {opt_acc_clean_str} \\\\"
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
        results[tv_name]["-1.0"] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=-1.0)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
        base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name]["-1.0"]['test_results'] = base_test_results
        results[tv_name]["-1.0"]['train_results'] = base_train_split_results
        
        base_model = copy.deepcopy(model)

        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=tv,
            search_range=(-2.0, 0.0),
            dataset=dataset,
            num_classes=num_classes,
            device=device
        )
        
        results[tv_name][best_coef] = OrderedDict()
        results[tv_name][best_coef]['test_results'] = best_results
        
        tv.apply_to(base_model, scaling_coef=best_coef)
        
        after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][best_coef]['train_results'] = after_tv_metrics
        
        
    return results

def pt_ft_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")

    augmentations = None
    if cfg['dataset']['name'] == 'cifar10':
        augmentations = [
            transformsv2.RandomCrop(224, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    elif cfg['dataset']['name'] == 'cifar100':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    elif cfg['dataset']['name'] == 'fashion_mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    elif cfg['dataset']['name'] == 'clothing1M':
        augmentations = [
            transformsv2.RandomHorizontalFlip(),
        ]
    
    
    base_dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'], augmentations)
    base_model = model_factory.create_model(cfg['model'], num_classes)
    strategy = cfg['strategy']
    base_dataset.inject_noise(**strategy['noise']['pretraining'])
    
    if not outputs_dir.joinpath(f"{cfg_name}/gold/weights/model_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
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
            **cfg['trainer']['pretraining'],
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
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        experiment_name = f"{cfg_name}/pretrain"

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        
        trainer_cls = StandardTrainer
        if strategy['finetuning_set'] == 'LowLoss': trainer_cls = TrainerRLS
        
        trainer = trainer_cls(
            outputs_dir=outputs_dir,
            **cfg['trainer']['pretraining'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        if strategy['finetuning_set'] == 'LowLoss':
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


    # if not outputs_dir.joinpath(f"{cfg_name}/finetune_gold/weights/model_weights.pth").exists():
    #     dataset = copy.deepcopy(base_dataset)
    #     model = copy.deepcopy(base_model)
        
    #     base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
    #     checkpoint = torch.load(base_model_ckp_path)
    #     model.load_state_dict(checkpoint)
        
    #     clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    #     dataset.set_trainset(clean_set, shuffle=True)
            
    #     experiment_name = f"{cfg_name}/finetune_gold"
    #     experiment_dir = outputs_dir / Path(experiment_name)

    #     weights_dir = experiment_dir / Path("weights")
    #     weights_dir.mkdir(exist_ok=True, parents=True)

    #     plots_dir = experiment_dir / Path("plots")
    #     plots_dir.mkdir(exist_ok=True, parents=True)
        
        

    #     trainer = StandardTrainer(
    #         outputs_dir=outputs_dir,
    #         **cfg['trainer']['pretraining'],
    #         exp_name=experiment_name,
    #         exp_tags=None,
    #     )
        
    #     results = trainer.fit(model, dataset, resume=False)

    #     torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

    #     class_names = [f"Class {i}" for i in range(num_classes)]
    #     confmat = trainer.confmat("Test")
    #     misc_utils.plot_confusion_matrix(
    #         cm=confmat,
    #         class_names=class_names,
    #         filepath=str(plots_dir / Path("confmat.png")),
    #         show=False,
    #     )
        
        
    # if not outputs_dir.joinpath(f"{cfg_name}/finetune_gt_noise/weights/model_weights.pth").exists():
    #     dataset = copy.deepcopy(base_dataset)
    #     model = copy.deepcopy(base_model)
        
    #     base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
    #     checkpoint = torch.load(base_model_ckp_path)
    #     model.load_state_dict(checkpoint)
        
    #     clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    #     dataset.set_trainset(noisy_set, shuffle=True)
            
    #     experiment_name = f"{cfg_name}/finetune_gt_noise"
    #     experiment_dir = outputs_dir / Path(experiment_name)

    #     weights_dir = experiment_dir / Path("weights")
    #     weights_dir.mkdir(exist_ok=True, parents=True)

    #     plots_dir = experiment_dir / Path("plots")
    #     plots_dir.mkdir(exist_ok=True, parents=True)
        
        

    #     trainer = StandardTrainer(
    #         outputs_dir=outputs_dir,
    #         **cfg['trainer']['finetuning'],
    #         exp_name=experiment_name,
    #         exp_tags=None,
    #     )
        
    #     results = trainer.fit(model, dataset, resume=False)

    #     torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

    #     class_names = [f"Class {i}" for i in range(num_classes)]
    #     confmat = trainer.confmat("Test")
    #     misc_utils.plot_confusion_matrix(
    #         cm=confmat,
    #         class_names=class_names,
    #         filepath=str(plots_dir / Path("confmat.png")),
    #         show=False,
    #     )

    
    for idx, noise_tv in enumerate(strategy['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
            checkpoint = torch.load(base_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout' or strategy['finetuning_set'] == 'Heldout+Train':
                if noise_tv['noise_type'] == 'T_matrix':
                    cm = get_confusion_matrix(
                        model,
                        dataset.get_num_classes(),
                        dataset.get_heldout_dataloader(),
                        trainer_utils.get_gpu_device()
                    )
                    T = estimate_T_from_confusion(cm, alpha=0.01)
                    noise_tv['T_mat'] = T
                
                
                dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
                dataset.inject_noise(**noise_tv)
                
                if noise_tv['noise_type'] == 'asymmetric':
                    finetune_only_noisy = strategy.get('finetune_only_noisy', False)
                    if finetune_only_noisy:
                        trainset = dataset.get_trainset()
                        noisy_indices = []
                        for item in trainset:
                            if len(item) == 4:
                                _, _, idx, is_noisy = item
                                if is_noisy:
                                    noisy_indices.append(idx)
                            else:
                                raise RuntimeError('The chosen dataset is not noisy!')
                        only_noisy_trainset = Subset(trainset, noisy_indices)
                        dataset.set_trainset(only_noisy_trainset, shuffle=True)
                        
                if strategy['finetuning_set'] == 'Heldout+Train':
                    orig_trainset = base_dataset.get_trainset()
                    print("Original Train Set Len :", len(orig_trainset))
                    heldout_trainset = dataset.get_trainset()
                    print("Heldout Train Set Len :", len(heldout_trainset))
                    merged_traineset = ConcatDataset([orig_trainset, heldout_trainset])
                    print("Merged Train Set Len :", len(merged_traineset))
                    dataset.set_trainset(merged_traineset, shuffle=True)
            
                    
            elif strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{strategy['percentage']:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
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
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'])
    
    strategy = cfg['strategy']
    dataset.inject_noise(**strategy['noise']['pretraining'])
    
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_dirs = {}
    results_dirs['cms'] = results_dir / 'confusion_mats'
    results_dirs['Ts'] = results_dir / 'transition_mats'
    results_dirs['W_norms'] = results_dir / 'weight_norms'
    results_dirs['TV_norms'] = results_dir / 'TV_norms'
    results_dirs['metrics'] = results_dir / 'metrics'    
    for dir in results_dirs.values():
        dir.mkdir(exist_ok=True, parents=True)

    
    
    
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
    # ft_gold_wieghts = torch.load(ft_gold_dir / 'weights/model_weights.pth', map_location=cpu)
    # ft_gt_noise_weights = torch.load(ft_gt_noise_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_dirs.items():
        finetune_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
 
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'Pretrain': pretrain_weights,
    #         'Gold': gold_weights,
    #         'FT Noise': next(iter(finetune_weights.items()))[1]
    #         },
    #     saving_path=results_dirs['W_norms'] / 'L1_pt_gold_ftnoise.png'
    # )
    
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'Pretrain': pretrain_weights,
    #         'FT Gold': ft_gold_wieghts,
    #         'FT Noise': next(iter(finetune_weights.items()))[1]
    #         },
    #     saving_path=results_dirs['W_norms'] / 'L1_pt_ftgold_ftnoise.png'
    # )
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'FT Gold': ft_gold_wieghts,
    #         'FT Noise': next(iter(finetune_weights.items()))[1]
    #         },
    #     saving_path=results_dirs['W_norms'] / 'L1_ftgold_ftnoise.png'
    # )
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'Gold': gold_weights,
    #         'Pretrain': pretrain_weights,
    #         'FT Gold': ft_gold_wieghts,
    #         },
    #     saving_path=results_dirs['W_norms'] / 'L1_pt_gold_ftgold.png'
    # )
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'Pretrain': pretrain_weights,
    #         'FT Gold': ft_gold_wieghts,
    #         'FT Noise': next(iter(finetune_weights.items()))[1],
    #         'FT GT Noise': ft_gt_noise_weights
    #         },
    #     saving_path=results_dirs['W_norms'] / 'L1_pt_ftgold_ftnoise_gtnoise.png'
    # )
    
    
    # ft_gold_tv = TaskVector(pretrain_weights, ft_gold_wieghts)
    # ft_gt_noise_tv = TaskVector(pretrain_weights, ft_gt_noise_weights)

    finetune_tvs = OrderedDict()
    
    for ft_expr, ft_weight in finetune_weights.items():
        finetune_tvs[f"{float(ft_expr.split('_')[0])*100:.0f}% Noise, {ft_expr.split('_')[1]} Seed"] = TaskVector(pretrain_weights, ft_weight)
        
    if len(finetune_tvs) == 1:
        finetune_tvs['Average TV'] = list(finetune_tvs.items())[0][1]
        finetune_tvs.popitem(last=False)
    else:
        finetune_tvs['Average TV'] = TaskVector.mean(finetune_tvs)
    finetune_tvs['Average TV Pruned 0.4'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.4)
    finetune_tvs['Average TV Pruned 0.6'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.6)
    finetune_tvs['Average TV Pruned 0.8'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.8)
    finetune_tvs['Average TV Pruned 0.9'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.9)
    finetune_tvs['Average TV Pruned 0.95'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.95)
    finetune_tvs['Average TV Pruned 0.99'] = finetune_tvs['Average TV'].prune_small_weights(rate=0.99)
    finetune_tvs['Random Vector'] = finetune_tvs['Average TV'].generate_random_vector_with_same_layer_norms(seed=11)
    # finetune_tvs['Gold'] = ft_gold_tv
    # finetune_tvs['Ground Truth Noise'] = ft_gt_noise_tv
    # finetune_tvs.move_to_end('Ground Truth Noise', last=False)
    # finetune_tvs.move_to_end('Gold', last=False)
    
    

    ft_tvs_list = list(finetune_tvs.values())
    print(finetune_tvs.keys())
    tv_names = list(finetune_tvs.keys())
    
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
    
    
    # weight_norm_analysis.plot_abs_weight_norms_compare(
    #     state_dicts={
    #         'Average TV': finetune_tvs['Average TV'].vector,
    #         'Gold TV': finetune_tvs['Gold'].vector,
    #         # 'Average TV Pruned 0.8': finetune_tvs['Average TV Pruned 0.8'].vector
    #         },
    #     saving_path=results_dirs['TV_norms'] / 'L1_norms.png'
    # )
    
    
    # weight_norm_analysis.plot_l2_weight_norms_compare(
    #     state_dicts={
    #         'Average TV': finetune_tvs['Average TV'].vector,
    #         'Gold TV': finetune_tvs['Gold'].vector,
    #         # 'Average TV Pruned 0.8': finetune_tvs['Average TV Pruned 0.8'].vector
    #         },
    #     saving_path=results_dirs['TV_norms'] / 'L2_norms.png'
    # )
    
    
    # base_model.load_state_dict(pretrain_weights)

    # cm_pt = get_confusion_matrix(
    #     base_model,
    #     dataset.get_num_classes(),
    #     dataset.get_heldout_dataloader(),
    #     gpu
    # )
    
    # T = estimate_T_from_confusion(cm_pt, alpha=0.01, lam=0.1)
    
    # misc_utils.plot_confusion_matrix(
    #     title='Noise Transition Matrix',
    #     cm=T,
    #     class_names=dataset.get_class_names(),
    #     color_map='vlag',
    #     color_bar=True,
    #     # vmin= 0.0,
    #     # vmax= 1.0,
    #     x_label='Classes',
    #     y_label='Classes',
    #     tick_label_font_size=6,
    #     filepath=results_dirs['Ts'] / 'transition_matrix.png',
    #     show=False
    # )
    
    # is_sym, kl = symmetric_noise_detected(T, kl_thresh=0.03)
    # if is_sym:
    #     print("Pattern is near-symmetric; using uniform off-diagonal.")
    
    
    
    # misc_utils.plot_confusion_matrix(
    #     title='Normalized Confusion Matrix',
    #     cm=row_normalize(cm_pt),
    #     class_names=dataset.get_class_names(),
    #     color_map='vlag',
    #     color_bar=True,
    #     # vmin= 0.0,
    #     # vmax= 1.0,
    #     x_label='Classes',
    #     y_label='Classes',
    #     tick_label_font_size=6,
    #     filepath=results_dirs['cms'] / 'pretrained_normalized.png',
    #     show=False
    # )
    
    
    # ft_model = copy.deepcopy(base_model)
    # ft_model.load_state_dict(next(iter(finetune_weights.items()))[1])
    # cm_ft = get_confusion_matrix(
    #     ft_model,
    #     dataset.get_num_classes(),
    #     dataset.get_heldout_dataloader(),
    #     gpu
    # )
    
    # misc_utils.plot_confusion_matrix(
    #     title='Normalized Confusion Matrix',
    #     cm=row_normalize(cm_ft),
    #     class_names=dataset.get_class_names(),
    #     color_map='vlag',
    #     color_bar=True,
    #     # vmin= 0.0,
    #     # vmax= 1.0,
    #     x_label='Classes',
    #     y_label='Classes',
    #     tick_label_font_size=6,
    #     filepath=results_dirs['cms'] / 'ft_noise_normalized.png',
    #     show=False
    # )
    
    # finetune_tvs['Average TV'].apply_to(base_model, scaling_coef=-1.0)
    # cm_ng = get_confusion_matrix(
    #     base_model,
    #     dataset.get_num_classes(),
    #     dataset.get_heldout_dataloader(),
    #     gpu
    # )
    
    # misc_utils.plot_confusion_matrix(
    #     title='Normalized Confusion Matrix',
    #     cm=row_normalize(cm_ng),
    #     class_names=dataset.get_class_names(),
    #     color_map='vlag',
    #     color_bar=True,
    #     # vmin= 0.0,
    #     # vmax= 1.0,
    #     x_label='Classes',
    #     y_label='Classes',
    #     tick_label_font_size=6,
    #     filepath=results_dirs['cms'] / 'negated_normalized.png',
    #     show=False
    # )
        

    # rank_dict = OrderedDict()
    # for tv_name, tv in finetune_tvs.items():
    #     rank_dict[tv_name] = tv.get_layer_rank()
        
    # with open(results_dir / 'ranks.json' , 'w') as json_file:
    #     json.dump(rank_dict, json_file, indent=4)
    
    
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
    
    # base_model.load_state_dict(next(iter(finetune_weights.items()))[1])
    # temp_res = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    # print(temp_res)
    # exit()
    
    base_model.load_state_dict(pretrain_weights)
    pt_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    pt_train_results = eval_model_on_clean_noise_splits(base_model, None, dataset, gpu)
    
    base_model.load_state_dict(gold_weights)
    gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    gold_train_results = eval_model_on_clean_noise_splits(base_model, None, dataset, gpu)
    
    # base_model.load_state_dict(ft_gold_wieghts)
    # ft_gold_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
    # ft_gold_train_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, gpu)
    
    base_model.load_state_dict(pretrain_weights)
    base_model.to(cpu)
    
    results_dict = OrderedDict()
    
    results_dict['Pretrain'] = {'test_results': pt_test_results, 'train_results': pt_train_results}
    results_dict['Gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
    # results_dict['Finetune Gold'] = {'test_results': ft_gold_test_results, 'train_results': ft_gold_train_results}
    # results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names[1:], ft_tvs_list[1:])), results_dict, cfg, dataset, num_classes, gpu)
    # results_dict = eval_model_on_tvs(base_model, OrderedDict(zip(tv_names, ft_tvs_list)), results_dict, cfg, dataset, num_classes, gpu)
    
    
    # print(results_dict)
    
    for alpha in tqdm(np.linspace(-0.1, -1.0, 10)):
    # for alpha in tqdm(np.linspace(-1.1, -1.6, 5)):
    
        base_model.load_state_dict(pretrain_weights, strict=False)
        finetune_tvs['Average TV'].apply_to(base_model, scaling_coef=alpha, strict=False)
        tv_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
        tv_train_results = eval_model_on_clean_noise_splits(base_model, None, dataset, gpu)

        results_dict[f"Avg, alpha={alpha}"] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        
    for alpha in tqdm(np.linspace(-0.1, -1.0, 10)):
    # for alpha in tqdm(np.linspace(-1.1, -1.6, 5)):
    
        base_model.load_state_dict(pretrain_weights, strict=False)
        finetune_tvs['Average TV Pruned 0.8'].apply_to(base_model, scaling_coef=alpha, strict=False)
        tv_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), gpu)
        tv_train_results = eval_model_on_clean_noise_splits(base_model, None, dataset, gpu)

        results_dict[f"Avg 0.8, alpha={alpha}"] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    
        
    with open(results_dirs['metrics'] / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    # generate_latex_table_from_results(results_dict, results_dirs['metrics'] / 'results_tex.txt')
    
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
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")

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