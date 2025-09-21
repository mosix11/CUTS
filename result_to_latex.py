import argparse
import os
import dotenv
import yaml
import json
import re
from pathlib import Path
import numpy as np
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List


def _load_metrics(config_dir: Path) -> Optional[Dict[str, Any]]:
    fpath = config_dir / 'metrics.json'
    if fpath.exists():
        try:
            with open(fpath, "r") as json_file:
                return json.load(json_file, object_pairs_hook=OrderedDict)
        except Exception:
            return None
        
    return None

def _get_test_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["test_results"]["ACC"])
    except Exception:
        return None

def _get_train_clean_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["train_results"]["clean_set"]["ACC"])
    except Exception:
        return None
    
def _get_train_clean_destruction_rate(block: Dict[str, Any]) -> Optional[float]:
    return 1 - _get_train_clean_acc(block)

def _get_train_noisy_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["train_results"]["noisy_set"]["ACC"])
    except Exception:
        return None
    
def _get_train_noisy_forget_rate(block: Dict[str, Any]) -> Optional[float]:
    return 1 - _get_train_noisy_acc(block)

def _get_train_noisy_healing_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["train_results"]["healing_noise"]["ACC"])
    except Exception:
        return None


def _collect_alpha_metrics(metrics: Dict[str, Any]) -> Dict[float, Dict]:
    alpha_metrics: Dict[float, Dict] = OrderedDict()

    for k, v in metrics.items():
        try:
            alpha = float(k)
        except Exception:
            continue
        alpha = round(alpha, 2)
        alpha_metrics[alpha] = OrderedDict()
        
        alpha_metrics[alpha]['utility'] = _get_test_acc(v)
        alpha_metrics[alpha]['forget_rate'] = _get_train_noisy_forget_rate(v)
        alpha_metrics[alpha]['destruction_rate'] = _get_train_clean_destruction_rate(v)
        alpha_metrics[alpha]['healing_rate'] = _get_train_noisy_healing_acc(v)
        
    sorted_items = sorted(alpha_metrics.items(), key=lambda x: x[0], reverse=True)
    
    return OrderedDict(sorted_items)


def _collect_baseline_metrics(metrics: Dict[str, Any]) -> Dict[float, Dict]:
    baseline_metrics: Dict[float, Dict] = OrderedDict()
    
    baseline_metrics['mix'] = OrderedDict()
    baseline_metrics['clean'] = OrderedDict()
    baseline_metrics['rnd'] = OrderedDict()
    
    baseline_metrics['mix']['utility'] = _get_test_acc(metrics['Mix'])
    baseline_metrics['mix']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Mix'])
    baseline_metrics['mix']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Mix'])
    baseline_metrics['mix']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Mix'])
    metrics.pop("Mix")
    
    
    baseline_metrics['clean']['utility'] = _get_test_acc(metrics['Gold'])
    baseline_metrics['clean']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Gold'])
    baseline_metrics['clean']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Gold'])
    baseline_metrics['clean']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Gold'])
    metrics.pop("Gold")
    
    
    baseline_metrics['rnd']['utility'] = _get_test_acc(metrics['Random Vector'])
    baseline_metrics['rnd']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Random Vector'])
    baseline_metrics['rnd']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Random Vector'])
    baseline_metrics['rnd']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Random Vector'])
    metrics.pop("Random Vector")

    return baseline_metrics


def _get_alpha_star_utility(alpha_metrics: Dict[float, Dict]) -> float:
    best_alpha = None
    best_alpha_utility = -np.inf
    
    for alpha, metrics in alpha_metrics.items():
        if metrics['utility'] > best_alpha_utility:
            best_alpha = alpha
            best_alpha_utility = metrics['utility'] 
            
    return best_alpha
       

def _get_alpha_star_forgetting(alpha_metrics: Dict[float, Dict], threshold:float) -> float:
    best_alpha = None
    
    for alpha, metrics in alpha_metrics.items():
        if round(metrics['forget_rate'], 2) >= threshold:
            best_alpha = alpha
            break
    return best_alpha     


# format as percentage with 1 decimal place
def _fmt_perct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100.0 * x:.1f}"

def _fmt_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, str]:
    return {k: _fmt_perct(v) for k, v in metrics.items()}

def generate_clip_symmetric_noise_table(results_dir:Path, cfgmap:OrderedDict):
    dataset_order = ["MNIST", "CIFAR10", "CIFAR100"]
    dataset_forget_trsh ={
        'MNIST': 0.9,
        'CIFAR10': 0.9,
        'CIFAR100': 0.9
    }
    noise_levels = [10, 20, 40, 60, 80]


    row_theta_mix: Dict[str, List[str]]   = {ds: ["-"] * 5 for ds in dataset_order}
    row_theta_clean: Dict[str, List[str]] = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_star_u: Dict[str, List[str]]  = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_star_fr: Dict[str, List[str]]  = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_kNN: Dict[str, List['str']] = {ds: ["-"] * 5 for ds in dataset_order}
    row_random_vec: Dict[str, List['str']] = {ds: ["-"] * 5 for ds in dataset_order}

    # ---------- fill data ----------
    for ds in dataset_order:
        if ds not in cfgmap:
            continue
        for j, eta in enumerate(noise_levels):
            config = cfgmap[ds].get(eta)

            metrics = _load_metrics(results_dir/config)
            
            metrics.pop('FT HO Clean')
            metrics.pop('alpha_s4')
            alpha_KNN = metrics.pop('alpha_KNN')
        
            baseline_metrics = _collect_baseline_metrics(metrics)
            alpha_metrics = _collect_alpha_metrics(metrics)
            
            alpha_star_utility = _get_alpha_star_utility(alpha_metrics)
            alpha_star_forgetting = _get_alpha_star_forgetting(alpha_metrics, dataset_forget_trsh[ds])
            
            mix_metrics  = _fmt_metrics(baseline_metrics['mix'])
            clean_metrics = _fmt_metrics(baseline_metrics['clean'])
            rnd_metrics = _fmt_metrics(baseline_metrics['rnd'])
            
            alpha_KNN_metrics = _fmt_metrics(alpha_metrics[alpha_KNN])
            alpha_star_utility_metrics = _fmt_metrics(alpha_metrics[alpha_star_utility])
            alpha_star_forgetting_metrics = _fmt_metrics(alpha_metrics[alpha_star_forgetting])
            
            row_theta_mix[ds][j] = mix_metrics['utility']
            row_theta_clean[ds][j] = clean_metrics['utility']
            row_alpha_star_u[ds][j] = alpha_star_utility_metrics['utility']
            row_alpha_star_fr[ds][j] = alpha_star_forgetting_metrics['utility']
            row_alpha_kNN[ds][j] = alpha_KNN_metrics['utility']
            row_random_vec[ds][j] = rnd_metrics['utility']
            
    
            

    # ---------- render LaTeX ----------
    def row_line(label: str, values_by_ds: Dict[str, List[str]]) -> str:
        cells = []
        for ds in dataset_order:
            cells.extend(values_by_ds[ds])
        return f"{label} & " + " & ".join(cells) + r" \\"

    header = r"""\begin{table}[ht]
\centering
\caption{CLIP Model: Utility of oracle, baselines, and corrected models with different negation strength indicated by the value of $\alpha$ across noise levels $\eta$ on MNIST, CIFAR-10, CIFAR-100 with symmetric noise.}
\label{tab:utility_vs_noise_rate}
\scriptsize
% \renewcommand{\arraystretch}{1}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccccccccc}
\toprule
& \multicolumn{5}{c}{MNIST} & \multicolumn{5}{c}{CIFAR10} & \multicolumn{5}{c}{CIFAR100} \\
\cmidrule(lr){2-6} \cmidrule(lr){7-11} \cmidrule(lr){12-16} 
Model & 10\% & 20\% & 40\% & 60\% & 80\% & 10\% & 20\% & 40\% & 60\% & 80\% & 10\% & 20\% & 40\% & 60\% & 80\% \\
\midrule
"""

    body_lines = [
        row_line(r"$\theta_{\text{mix}}$", row_theta_mix),
        row_line(r"$\theta_{\text{clean}}$", row_theta_clean),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\alpha^\ast_a$", row_alpha_star_u),
        row_line(r"$\alpha^\ast_f$", row_alpha_star_fr),
        row_line(r"$\hat{\alpha}^\ast_{\text{kNN}}$", row_alpha_kNN),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\tau_{r}$", row_random_vec),
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer

    out_path = Path("clip_symmetric_noise_table.txt")
    out_path.write_text(table_tex)
    return out_path

if __name__ == "__main__":
    with open('configmap.yaml', 'r') as file:
        configmap = yaml.full_load(file)
    
    clip_models_cfgs = configmap['clip_models']
    clip_models_results_dir = Path('results/single_experiment/clip_noise_TA')
    
    clip_symmetric_cfgs = OrderedDict()
    clip_symmetric_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho_2']['symmetric']
    clip_symmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho_2']['symmetric']
    clip_symmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho_2']['symmetric']
    
    
    clip_asymmetric_cfgs = OrderedDict()
    clip_asymmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho_2']['asymmetric']
    clip_asymmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho_2']['asymmetric']
    
    clip_ic_cfgs = OrderedDict()
    clip_ic_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho_2']['ic']
    clip_ic_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho_2']['ic']
    clip_ic_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho_2']['ic']
    
    clip_poison_cfgs = OrderedDict()
    clip_poison_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho_2']['poison']
    clip_poison_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho_2']['poison']
    clip_poison_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho_2']['poison']
    
    
    generate_clip_symmetric_noise_table(clip_models_results_dir, clip_symmetric_cfgs)
    
    regular_models_cfgs = configmap['regular_models']
    clip_models_results_dir = Path('results/single_experiment/pretrain_on_noisy')
    
    
    