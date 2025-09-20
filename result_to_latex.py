import argparse
import os
import dotenv
import yaml
import json
import re
from pathlib import Path

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

def _get_train_healing_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["train_results"]["healing_noise"]["ACC"])
    except Exception:
        return None


def _collect_alpha_metrics(metrics: Dict[str, Any]) -> Dict[float, Dict]:
    alpha_metrics: Dict[float, Dict] = OrderedDict()
    for k, v in metrics.items():
        if k in {"Mix", "Gold", "FT HO Clean"}:
            continue
        try:
            alpha = float(k)
        except Exception:
            continue
        alpha = round(alpha, 2)
        alpha_metrics[alpha] = OrderedDict()
        
        alpha_metrics['utility'] = _get_test_acc(v)
        alpha_metrics['forget_rate'] = _get_train_noisy_forget_rate(v)
        alpha_metrics['destruction_rate'] = _get_train_clean_destruction_rate(v)
        alpha_metrics['healing_rate'] = _get_train_healing_acc(v)
        
    return alpha_metrics


def _collect_baseline_metrix(metrics: Dict[str, Any]) -> Dict[float, Dict]:
    baseline_metrics: Dict[float, Dict] = OrderedDict()
    
    baseline_metrics['mix'] = OrderedDict()
    baseline_metrics['clean'] = OrderedDict()
    
    baseline_metrics['mix']['utility'] = _get_test_acc(metrics['Mix'])
    baseline_metrics['mix']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Mix'])
    baseline_metrics['mix']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Mix'])
    baseline_metrics['mix']['healing_rate'] = _get_train_healing_acc(metrics['Mix'])
    
    
    baseline_metrics['clean']['utility'] = _get_test_acc(metrics['Gold'])
    baseline_metrics['clean']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Gold'])
    baseline_metrics['clean']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Gold'])
    baseline_metrics['clean']['healing_rate'] = _get_train_healing_acc(metrics['Gold'])

    return baseline_metrics

def generate_clip_symmetric_noise_table(results_dir:Path, cfgmap:OrderedDict):
    dataset_order = ["MNIST", "CIFAR10", "CIFAR100"]
    noise_levels = [10, 20, 40, 60, 80]

    # ---------- helpers ----------
    

    def collect_alpha_accs(metrics: Dict[str, Any]) -> List[Tuple[float, float]]:
        alpha_accs: List[Tuple[float, float]] = []
        for k, v in metrics.items():
            if k in {"Mix", "Gold", "FT HO Clean"}:
                continue
            try:
                alpha = float(k)
            except Exception:
                continue
            acc = get_test_acc(v)
            if acc is not None:
                alpha_accs.append((alpha, acc))
        return alpha_accs

    def get_acc_near_alpha(alpha_accs: List[Tuple[float, float]], target: float) -> Optional[float]:
        if not alpha_accs:
            return None
        alpha, acc = min(alpha_accs, key=lambda p: abs(p[0] - target))
        # tolerate float grid quirks; grid step is 0.05
        if abs(alpha - target) > 0.075:
            return None
        return acc

    def get_best_alpha_acc(alpha_accs: List[Tuple[float, float]]) -> Optional[float]:
        if not alpha_accs:
            return None
        return max(alpha_accs, key=lambda p: p[1])[1]

    # format as percentage with 1 decimal place
    def fmt(x: Optional[float]) -> str:
        return "-" if x is None else f"{100.0 * x:.1f}"

    # ---------- row holders ----------
    row_theta_mix: Dict[str, List[str]]   = {ds: ["-"] * 5 for ds in dataset_order}
    row_theta_clean: Dict[str, List[str]] = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_05: Dict[str, List[str]]    = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_10: Dict[str, List[str]]    = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_15: Dict[str, List[str]]    = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_20: Dict[str, List[str]]    = {ds: ["-"] * 5 for ds in dataset_order}
    row_alpha_star: Dict[str, List[str]]  = {ds: ["-"] * 5 for ds in dataset_order}

    # ---------- fill data ----------
    for ds in dataset_order:
        if ds not in cfgmap:
            continue
        for j, eta in enumerate(noise_levels):
            config = cfgmap[ds].get(eta)
            if not config:
                continue

            metrics = load_metrics(results_dir/config)
            if not metrics:
                continue

            mix_acc  = get_test_acc(metrics.get("Mix", {}))
            gold_acc = get_test_acc(metrics.get("Gold", {}))

            row_theta_mix[ds][j]   = fmt(mix_acc)
            row_theta_clean[ds][j] = fmt(gold_acc)

            alpha_accs = collect_alpha_accs(metrics)

            acc_05 = get_acc_near_alpha(alpha_accs, target=-0.5)
            acc_10 = get_acc_near_alpha(alpha_accs, target=-1.0)
            acc_15 = get_acc_near_alpha(alpha_accs, target=-1.5)
            acc_20 = get_acc_near_alpha(alpha_accs, target=-2.0)

            row_alpha_05[ds][j] = fmt(acc_05)
            row_alpha_10[ds][j] = fmt(acc_10)
            row_alpha_15[ds][j] = fmt(acc_15)
            row_alpha_20[ds][j] = fmt(acc_20)

            best_acc = get_best_alpha_acc(alpha_accs)
            row_alpha_star[ds][j] = fmt(best_acc)

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
% \begin{tabular}{l|ccccc|ccccc|ccccc}
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
        row_line(r"$\alpha=0.5$", row_alpha_05),
        row_line(r"$\alpha=1$", row_alpha_10),
        row_line(r"$\alpha=1.5$", row_alpha_15),
        row_line(r"$\alpha=2$", row_alpha_20),
        row_line(r"$\alpha^\ast$", row_alpha_star),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\tau_{\text{random}}$", {ds: ["-"] * 5 for ds in dataset_order}),
        row_line(r"GA", {ds: ["-"] * 5 for ds in dataset_order}),
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
    clip_symmetric_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho2']['symmetric']
    clip_symmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho2']['symmetric']
    clip_symmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho2']['symmetric']
    
    
    clip_asymmetric_cfgs = OrderedDict()
    clip_asymmetric_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho2']['asymmetric']
    clip_asymmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho2']['asymmetric']
    clip_asymmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho2']['asymmetric']
    
    clip_ic_cfgs = OrderedDict()
    clip_ic_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho2']['ic']
    clip_ic_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho2']['ic']
    clip_ic_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho2']['ic']
    
    clip_poison_cfgs = OrderedDict()
    clip_poison_cfgs['MNIST'] = clip_models_cfgs['MNIST']['ho2']['poison']
    clip_poison_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['ho2']['poison']
    clip_poison_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['ho2']['poison']
    
    
    generate_clip_symmetric_noise_table(clip_models_results_dir, clip_symmetric_cfgs)
    
    regular_models_cfgs = configmap['regular_models']
    clip_models_results_dir = Path('results/single_experiment/pretrain_on_noisy')
    
    
    