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

from typing import OrderedDict as _OrderedDictType
import matplotlib.pyplot as plt


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
    
def _get_ho_acc(block: Dict[str, Any]) -> Optional[float]:
    try:
        return float(block["ho_results"]["ACC"])
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
        
        if 'ho_results' in v:
            ho_utility = _get_ho_acc(v)
            alpha_metrics[alpha]['ho_utility'] = ho_utility
            alpha_metrics[alpha]['ho_forget_rate'] = 1 - ho_utility
        
    sorted_items = sorted(alpha_metrics.items(), key=lambda x: x[0], reverse=True)
    
    return OrderedDict(sorted_items)


def _collect_baseline_metrics(metrics: Dict[str, Any]) -> Dict[float, Dict]:
    baseline_metrics: Dict[float, Dict] = OrderedDict()
    
    baseline_metrics['mix'] = OrderedDict()
    baseline_metrics['clean'] = OrderedDict()
    baseline_metrics['rnd'] = OrderedDict()
    
    baseline_metrics['mix']['utility'] = _get_test_acc(metrics['Mix'])
    # baseline_metrics['mix']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Mix'])
    # baseline_metrics['mix']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Mix'])
    # baseline_metrics['mix']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Mix'])
    
    if 'ho_results' in metrics['Mix']:
        ho_utility = _get_ho_acc(metrics['Mix'])
        baseline_metrics['mix']['ho_utility'] = ho_utility
        baseline_metrics['mix']['ho_forget_rate'] = 1 - ho_utility
    metrics.pop("Mix")
    
    
    baseline_metrics['clean']['utility'] = _get_test_acc(metrics['Gold'])
    # baseline_metrics['clean']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Gold'])
    # baseline_metrics['clean']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Gold'])
    # baseline_metrics['clean']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Gold'])
    if 'ho_results' in metrics['Gold']:
        ho_utility = _get_ho_acc(metrics['Gold'])
        baseline_metrics['clean']['ho_utility'] = ho_utility
        baseline_metrics['clean']['ho_forget_rate'] = 1 - ho_utility
    metrics.pop("Gold")
    
    
    baseline_metrics['rnd']['utility'] = _get_test_acc(metrics['Random Vector'])
    baseline_metrics['rnd']['forget_rate'] = _get_train_noisy_forget_rate(metrics['Random Vector'])
    baseline_metrics['rnd']['destruction_rate'] = _get_train_clean_destruction_rate(metrics['Random Vector'])
    baseline_metrics['rnd']['healing_rate'] = _get_train_noisy_healing_acc(metrics['Random Vector'])
    if 'ho_results' in metrics['Random Vector']:
        ho_utility = _get_ho_acc(metrics['Random Vector'])
        baseline_metrics['rnd']['ho_utility'] = ho_utility
        baseline_metrics['rnd']['ho_forget_rate'] = 1 - ho_utility
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
    if best_alpha == None:
        print('No forgetting alpha was found, returning the biggest alpha.')
        best_alpha = list(alpha_metrics.keys())[-1]
    return best_alpha     


def _get_alpha_max_healing(alpha_metrics: Dict[float, Dict]) -> float:
    best_alpha = None
    max_healing = -np.inf
    for alpha, metrics in alpha_metrics.items():
        if metrics['healing_rate'] >= max_healing:
            best_alpha = alpha
            max_healing = metrics['healing_rate']
    return best_alpha     


# format as percentage with 1 decimal place
def _fmt_perct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100.0 * round(x, 3):.1f}"

def _fmt_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, str]:
    return {k: _fmt_perct(v) for k, v in metrics.items()}

def generate_clip_noise_utlity_table(
    results_dir:Path,
    cfgmap:OrderedDict,
    dataset_order:List[str] = ["MNIST", "CIFAR10", "CIFAR100"],
    dataset_forget_trsh:Dict[str, float] ={
        'MNIST': 0.9,
        'CIFAR10': 0.9,
        'CIFAR100': 0.9
    },
    noise_levels:List[int] = [10, 20, 40, 60, 80],
    outputfile_path:Path = Path("./visulaization_dir/clip_symmetric_noise_table.txt")
    ):


    row_theta_mix: Dict[str, List[str]]   = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_theta_clean: Dict[str, List[str]] = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_alpha_star_u: Dict[str, List[str]]  = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_alpha_star_fr: Dict[str, List[str]]  = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_alpha_kNN: Dict[str, List['str']] = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_random_vec: Dict[str, List['str']] = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    row_recovery_kNN: Dict[str, List['str']] = {ds: ["-"] * len(noise_levels) for ds in dataset_order}
    
    # ---------- fill data ----------
    for ds in dataset_order:
        if ds not in cfgmap:
            continue
        for j, eta in enumerate(noise_levels):
            config = cfgmap[ds].get(eta)

            metrics = _load_metrics(results_dir/config)
            
            metrics.pop('FT HO Clean', None)
            metrics.pop('alpha_s4', None)
            alpha_KNN = metrics.pop('alpha_KNN')
        
            baseline_metrics = _collect_baseline_metrics(metrics)
            alpha_metrics = _collect_alpha_metrics(metrics)
            
            alpha_star_utility = _get_alpha_star_utility(alpha_metrics)
            alpha_star_forgetting = _get_alpha_star_forgetting(alpha_metrics, dataset_forget_trsh[ds])
            recovery_kNN = (alpha_metrics[alpha_KNN]['utility'] - baseline_metrics['mix']['utility'])/(baseline_metrics['clean']['utility'] - baseline_metrics['mix']['utility'])
            recovery_kNN = _fmt_perct(recovery_kNN)
            
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
            row_recovery_kNN[ds][j] = recovery_kNN
    
            

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
        row_line(r"$\tau_{r}$", row_random_vec),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\alpha^\ast_a$", row_alpha_star_u),
        row_line(r"$\alpha^\ast_f$", row_alpha_star_fr),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\hat{\alpha}^\ast_{\text{kNN}}$", row_alpha_kNN),
        row_line(r"recovery", row_recovery_kNN),
        
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer

    outputfile_path.write_text(table_tex)
    return outputfile_path


def generate_clip_noise_fr_dr_hr_table(
    results_dir:Path,
    cfgmap:OrderedDict,
    forget_threshold: float = 0.89,
    noise_levels:List[int] = [10, 20, 40, 60, 80],
    outputfile_path:Path = Path("./visulaization_dir/clip_asymmetric_noise_fr_dr_hr_table.txt")  
):
    """
    Build a CIFAR-10 table showing, for each noise level, the metrics at three alpha choices:
      - alpha*_a (best utility)
      - alpha*_f (first alpha meeting forgetting threshold)
      - alpha_hat*_knn (kNN-estimated alpha)
    Rows: UT, RR, FR, HR, DR
    """

    # Allocate 15 columns (5 noise levels × 3 alphas each)
    ut_vals = ["-"] * len(noise_levels)*3
    rr_vals = ["-"] * len(noise_levels)*3
    fr_vals = ["-"] * len(noise_levels)*3
    hr_vals = ["-"] * len(noise_levels)*3
    dr_vals = ["-"] * len(noise_levels)*3

    for i, eta in enumerate(noise_levels):
        config = cfgmap.get(eta)
        if not config:
            continue

        metrics = _load_metrics(results_dir / config)
        if not metrics:
            continue

        # Clean up non-alpha keys that appear in files
        metrics.pop("FT HO Clean", None)
        metrics.pop("alpha_s4", None)
        alpha_KNN = metrics.pop("alpha_KNN", None)
        if alpha_KNN is not None:
            try:
                alpha_KNN = round(float(alpha_KNN), 2)
            except Exception:
                alpha_KNN = None

        # Baselines and alpha grid
        baseline_metrics = _collect_baseline_metrics(metrics)  # pops Mix/Gold/RV out of `metrics`
        alpha_metrics = _collect_alpha_metrics(metrics)        # keys are rounded to 2 decimals

        # Resolve alphas
        alpha_a = _get_alpha_star_utility(alpha_metrics)  # best utility alpha
        alpha_f = _get_alpha_star_forgetting(alpha_metrics, forget_threshold)

        alpha_list = [alpha_a, alpha_f, alpha_KNN]

        # For each of the 3 alpha choices, fill 1 cell per row
        for j, alpha in enumerate(alpha_list):
            col_idx = i * 3 + j  # position within the 15 columns

            if alpha is None or alpha not in alpha_metrics:
                # leave "-" if alpha missing/unavailable
                continue

            a_metrics = alpha_metrics[alpha]
            ut = a_metrics.get("utility", None)
            fr = a_metrics.get("forget_rate", None)
            hr = a_metrics.get("healing_rate", None)
            dr = a_metrics.get("destruction_rate", None)

            # Recovery rate relative to mix→clean gap
            mix_ut = baseline_metrics["mix"]["utility"]
            clean_ut = baseline_metrics["clean"]["utility"]
            denom = (clean_ut - mix_ut) if (mix_ut is not None and clean_ut is not None) else None
            rr = None
            if ut is not None and denom is not None and abs(denom) > 1e-12:
                rr = (ut - mix_ut) / denom

            ut_vals[col_idx] = _fmt_perct(ut)
            rr_vals[col_idx] = _fmt_perct(rr)
            fr_vals[col_idx] = _fmt_perct(fr)
            hr_vals[col_idx] = _fmt_perct(hr)
            dr_vals[col_idx] = _fmt_perct(dr)

    # ---------- render LaTeX ----------
    def row_line(label: str, values: list[str]) -> str:
        return f"{label} & " + " & ".join(values) + r" \\"

    header = r"""\begin{table}[ht]
\centering
\label{tab:clip_sym_utility_vs_alpha}
\scriptsize
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccccccccc}
\toprule
& \multicolumn{15}{c}{CIFAR10} \\
\cmidrule(lr){2-16}
& \multicolumn{3}{c}{10\%} & \multicolumn{3}{c}{20\%} & \multicolumn{3}{c}{40\%} & \multicolumn{3}{c}{60\%}  & \multicolumn{3}{c}{80\%}  \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}
Metric & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ \\
\midrule
"""
    body_lines = [
        row_line("UT", ut_vals),
        row_line("RR", rr_vals),
        row_line("FR", fr_vals),
        row_line("HR", hr_vals),
        row_line("DR", dr_vals),
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer
    outputfile_path.write_text(table_tex)
    return outputfile_path

def plot_alpha_interplay(
    results_dir: Path,
    config_rel_path: str,
    dataset_name: str = "CIFAR10",
    forget_threshold: float = 0.9,
    out_dir: Path = Path("./visulaization_dir"),
) -> Path:
    """
    Plot UT, FR, HR, DR vs alpha for a single experiment and highlight
    alpha*_a (best UT), alpha*_f (first FR >= threshold), and alpha_kNN.

    Ordering:
      - Preferred: use only nonpositive alphas (<=0), put 0 at left and
        more negative to the right via ax.set_xlim(0, min_negative).
      - Fallback: if no negatives exist, plot against |alpha| ascending.

    Styling:
      - No markers, no background grid.
      - Horizontal light-gray line for Clean UT (upper bound).
      - Mix metrics injected at alpha=0.
      - Alpha annotations rendered on the x-axis (bottom).
    """
    metrics = _load_metrics(results_dir / config_rel_path)
    if not metrics:
        raise FileNotFoundError(f"metrics.json not found or unreadable for {config_rel_path}")

    # clean out non-alpha entries; extract alpha_KNN
    metrics.pop("FT HO Clean", None)
    metrics.pop("alpha_s4", None)
    alpha_kNN = metrics.pop("alpha_KNN", None)
    try:
        alpha_kNN = None if alpha_kNN is None else round(float(alpha_kNN), 2)
    except Exception:
        alpha_kNN = None

    baselines = _collect_baseline_metrics(metrics)   # pops Mix/Gold/Random Vector
    alpha_grid = _collect_alpha_metrics(metrics)     # keys are rounded floats

    # Inject Mix at alpha=0
    mix_alpha = 0.0
    alpha_grid[mix_alpha] = {
        "utility": baselines["mix"]["utility"],
        "forget_rate": baselines["mix"]["forget_rate"],
        "healing_rate": baselines["mix"]["healing_rate"],
        "destruction_rate": baselines["mix"]["destruction_rate"],
    }

    # Resolve α*’s
    alpha_star_u = _get_alpha_star_utility(alpha_grid)
    alpha_star_f = _get_alpha_star_forgetting(alpha_grid, forget_threshold)

    # --------- Choose x mapping & ordering ---------
    all_as = list(alpha_grid.keys())
    neg_as = sorted([a for a in all_as if a < 0], key=lambda a: abs(a))  # -0.01, -0.02, ...
    use_abs_mode = (len(neg_as) == 0)  # fallback if there are no negative alphas

    if not use_abs_mode:
        # Preferred mode: x = alpha for a <= 0, with 0 on the left and more negative to the right
        ordered_alphas: List[float] = [mix_alpha] + neg_as  # 0, -small, -bigger, ...
        x_vals = ordered_alphas  # x is the actual alpha
        xlabel = r"$\alpha$"
        # Helper to pull series in this order
        def _series(key: str):
            return [alpha_grid[a].get(key, None) for a in ordered_alphas]
        # Annotation x-positions (only show if a<=0 and exists)
        def _ann_x(a: Optional[float]) -> Optional[float]:
            return a if (a is not None and a in alpha_grid and a <= 0.0) else None
        x_left, x_right = 0.0, (min(neg_as) if neg_as else 0.0)  # invert axis
    else:
        # Fallback: x = |alpha| ascending
        ordered_alphas = sorted(all_as, key=lambda a: (abs(a), a))
        x_vals = [abs(a) for a in ordered_alphas]
        xlabel = r"$|\alpha|$"
        def _series(key: str):
            return [alpha_grid[a].get(key, None) for a in ordered_alphas]
        def _ann_x(a: Optional[float]) -> Optional[float]:
            return abs(a) if (a is not None and a in alpha_grid) else None
        x_left, x_right = 0.0, (max(x_vals) if x_vals else 0.0)

    UT = _series("utility")
    FR = _series("forget_rate")
    HR = _series("healing_rate")
    DR = _series("destruction_rate")

    # ---------- plotting ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = re.sub(r"[^\w\-]+", "_", str(config_rel_path))
    save_path = out_dir / f"interplay_{cfg_name}.png"

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    # Lines only (no markers)
    ax.plot(x_vals, UT, linewidth=2.2, label="UT")
    ax.plot(x_vals, FR, linewidth=2.0, label="FR")
    ax.plot(x_vals, HR, linewidth=2.0, label="HR")
    ax.plot(x_vals, DR, linewidth=2.0, label="DR")

    # Axes styling
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=10)

    # Invert x-axis direction if using preferred (alpha<=0) mode
    if not use_abs_mode:
        ax.set_xlim(x_left, x_right)  # e.g., (0.0, -0.6) so left=0 → right=more negative
    else:
        ax.set_xlim(x_left, x_right)

    # Title
    ax.set_title(f"{dataset_name}: UT/FR/HR/DR vs. {xlabel}", fontsize=12)

    # Horizontal reference line at Clean UT (upper bound)
    clean_ut = baselines["clean"]["utility"]
    if clean_ut is not None:
        ax.axhline(clean_ut, color="#BBBBBB", linewidth=1.5, linestyle="-", alpha=0.9, label="Clean UT (upper bound)")

    # Alpha annotations on x-axis (bottom)
    def _annotate_on_axis(x_at: Optional[float], label: str, linestyle, y_margin = 0.015):
        if x_at is None:
            return
        ymin, ymax = ax.get_ylim()
        ax.axvline(x_at, linestyle=linestyle, linewidth=1.1, alpha=0.6, color="black")
        ax.text(
            x_at,
            ymin + y_margin * (ymax - ymin),
            label,
            va="bottom",
            ha="center",
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
            clip_on=False,
        )

    _annotate_on_axis(_ann_x(alpha_star_u), r"$\alpha^\ast_a$", linestyle=':')
    _annotate_on_axis(_ann_x(alpha_star_f), r"$\alpha^\ast_f$", linestyle="--", y_margin=0.1)
    _annotate_on_axis(_ann_x(alpha_kNN),    r"$\hat{\alpha}^\ast_{\mathrm{kNN}}$", linestyle="-.")

    # Legend
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return save_path


def plot_noise_alpha_interplay_dual(
    results_dir: Path,
    config_rel_path_A: str,
    config_rel_path_B: str,
    dataset_name_A: str = "CIFAR10",
    dataset_name_B: str = "CIFAR10",
    forget_threshold_A: float = 0.9,
    forget_threshold_B: float = 0.9,
    out_dir: Path = Path("./visulaization_dir"),
) -> Path:
    """
    Make a 1x2 figure of UT/FR/HR/DR vs α for two experiments (A, B), with:
      - Preferred x-ordering: only nonpositive alphas (<=0), 0 at left, more negative to the right.
      - Fallback: if no negatives exist, plot against |alpha| ascending.
      - Mix injected at α=0.
      - Clean UT horizontal reference line.
      - α*_a, α*_f, α_kNN highlighted on x-axis (bottom labels).
      - Shared y-axis and a single, shared legend.

    Saves to: ./visulaization_dir/interplay_{nameA}__{nameB}.png
    Returns: saved Path
    """

    def _prepare_one(
        config_rel_path: str,
        dataset_name: str,
        forget_threshold: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load metrics, inject mix@0, compute alphas, choose ordering, return plotting payload + meta."""
        metrics = _load_metrics(results_dir / config_rel_path)
        if not metrics:
            raise FileNotFoundError(f"metrics.json not found or unreadable for {config_rel_path}")

        # clean out non-alpha entries; extract alpha_KNN
        metrics.pop("FT HO Clean", None)
        metrics.pop("alpha_s4", None)
        alpha_kNN = metrics.pop("alpha_KNN", None)
        try:
            alpha_kNN = None if alpha_kNN is None else round(float(alpha_kNN), 2)
        except Exception:
            alpha_kNN = None

        baselines = _collect_baseline_metrics(metrics)   # pops Mix/Gold/Random Vector
        alpha_grid = _collect_alpha_metrics(metrics)     # keys are rounded floats

        # Inject Mix at α=0
        mix_alpha = 0.0
        alpha_grid[mix_alpha] = {
            "utility": baselines["mix"]["utility"],
            "forget_rate": baselines["mix"]["forget_rate"],
            "healing_rate": baselines["mix"]["healing_rate"],
            "destruction_rate": baselines["mix"]["destruction_rate"],
        }

        # Resolve α*’s
        alpha_star_u = _get_alpha_star_utility(alpha_grid)
        alpha_star_f = _get_alpha_star_forgetting(alpha_grid, forget_threshold)

        # Choose x mapping & ordering
        all_as = list(alpha_grid.keys())
        neg_as = sorted([a for a in all_as if a < 0], key=lambda a: abs(a))  # -0.01, -0.02, ...
        use_abs_mode = (len(neg_as) == 0)

        if not use_abs_mode:
            ordered_alphas: List[float] = [mix_alpha] + neg_as
            x_vals = ordered_alphas
            xlabel = r"$\alpha$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return a if (a is not None and a in alpha_grid and a <= 0.0) else None
            x_left, x_right = 0.0, (min(neg_as) if neg_as else 0.0)  # invert axis
        else:
            ordered_alphas = sorted(all_as, key=lambda a: (abs(a), a))
            x_vals = [abs(a) for a in ordered_alphas]
            xlabel = r"$|\alpha|$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return abs(a) if (a is not None and a in alpha_grid) else None
            x_left, x_right = 0.0, (max(x_vals) if x_vals else 0.0)

        payload = dict(
            dataset_name=dataset_name,
            config_rel_path=config_rel_path,
            baselines=baselines,
            alpha_grid=alpha_grid,
            alpha_star_u=alpha_star_u,
            alpha_star_f=alpha_star_f,
            alpha_kNN=alpha_kNN,
            x_vals=x_vals,
            xlabel=xlabel,
            xlim=(x_left, x_right),
            UT=_series("utility"),
            FR=_series("forget_rate"),
            HR=_series("healing_rate"),
            DR=_series("destruction_rate"),
            ann_x=_ann_x,
            use_abs_mode=use_abs_mode,
        )
        meta = dict()  # reserved for future needs
        return payload, meta

    # Prepare both panels
    payloadA, _ = _prepare_one(config_rel_path_A, dataset_name_A, forget_threshold_A)
    payloadB, _ = _prepare_one(config_rel_path_B, dataset_name_B, forget_threshold_B)

    # ---------- plotting ----------
    nameA = re.sub(r"[^\w\-]+", "_", str(config_rel_path_A))
    nameB = re.sub(r"[^\w\-]+", "_", str(config_rel_path_B))
    save_path = out_dir / f"noise_interplay_{nameA}__{nameB}.png"

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=220, sharey=True)

    def _plot_panel(ax, P: Dict[str, Any], title: Optional[str] = None, add_legend: bool = False):
        # Lines only (no markers)
        h_ut, = ax.plot(P["x_vals"], P["UT"], linewidth=2.2, label="UT")
        h_fr, = ax.plot(P["x_vals"], P["FR"], linewidth=2.0, label="FR")
        h_hr, = ax.plot(P["x_vals"], P["HR"], linewidth=2.0, label="HR")
        h_dr, = ax.plot(P["x_vals"], P["DR"], linewidth=2.0, label="DR")

        # Axes styling
        ax.set_xlabel(P["xlabel"], fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(*P["xlim"])

        # Panel title
        if title is None:
            title = f"{P['dataset_name']} · {Path(P['config_rel_path']).parent.name}"
        ax.set_title(title, fontsize=12)

        # Clean UT reference
        clean_ut = P["baselines"]["clean"]["utility"]
        if clean_ut is not None:
            ax.axhline(clean_ut, color="#BBBBBB", linewidth=1.5, linestyle="--", alpha=0.9)

        # Alpha annotations on x-axis (bottom)
        def _annotate_on_axis(x_at: Optional[float], label: str, linestyle, y_margin=0.015):
            if x_at is None:
                return
            ymin, ymax = ax.get_ylim()
            ax.axvline(x_at, linestyle=linestyle, linewidth=1.1, alpha=0.6, color="black")
            ax.text(
                x_at,
                ymin + y_margin * (ymax - ymin),
                label,
                va="bottom",
                ha="center",
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
                clip_on=False,
            )

        _annotate_on_axis(P["ann_x"](P["alpha_star_u"]), r"$\alpha^\ast_a$", linestyle=":")
        _annotate_on_axis(P["ann_x"](P["alpha_star_f"]), r"$\alpha^\ast_f$", linestyle="--", y_margin=0.08)
        _annotate_on_axis(P["ann_x"](P["alpha_kNN"]),   r"$\hat{\alpha}^\ast_{\mathrm{kNN}}$", linestyle="-.")

        # return handles for a single shared legend if requested
        if add_legend:
            return [h_ut, h_fr, h_hr, h_dr]
        return []

    # Left panel: add legend handles
    handles = _plot_panel(axes[0], payloadA, title=f"{payloadA['dataset_name']}", add_legend=True)
    # Right panel
    _plot_panel(axes[1], payloadB, title=f"{payloadB['dataset_name']}")

    # Shared Y label on the figure
    fig.text(0.04, 0.5, "Metric value", va="center", rotation="vertical", fontsize=11)

    # Single shared legend (deduplicated by dict)
    by_label = {}
    for h in handles:
        by_label[h.get_label()] = h
    fig.legend(list(by_label.values()), list(by_label.keys()),
               loc="lower center", ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.52, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(left=0.09, bottom=0.18)  # increase left from default

    fig.subplots_adjust(bottom=0.18)  # make room for shared legend
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return save_path


def generate_clip_IC_utlity_table(
    results_dir:Path,
    cfgmap:OrderedDict,
    dataset_order:List[str] = ["MNIST", "CIFAR10", "CIFAR100"],
    outputfile_path:Path = Path("./visulaization_dir/clip_IC_noise_table.txt")
    ):
    corruption_classes = {
        "MNIST": ['1-7'],
        "CIFAR10": ['3-5', '6-9'],
        "CIFAR100": ['47-52', '0-24'],
    }


    row_theta_mix: Dict[str, List[str]]   = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_theta_clean: Dict[str, List[str]] = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_alpha_star_u: Dict[str, List[str]]  = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_alpha_star_h: Dict[str, List[str]]  = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_alpha_IC: Dict[str, List['str']] = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_random_vec: Dict[str, List['str']] = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    row_recovery_IC: Dict[str, List['str']] = {ds: ["-"] * len(corruption_classes[ds]) for ds in dataset_order}
    
    # ---------- fill data ----------
    for ds in dataset_order:
        if ds not in cfgmap:
            continue
        for j, class_maps in enumerate(corruption_classes[ds]):
            config = cfgmap[ds].get(class_maps)

            metrics = _load_metrics(results_dir/config)
            
            metrics.pop('FT HO Clean')
            alpha_IC = metrics.pop('alpha_IC')
            
        
            baseline_metrics = _collect_baseline_metrics(metrics)
            alpha_metrics = _collect_alpha_metrics(metrics)
            alpha_star_utility = _get_alpha_star_utility(alpha_metrics)
            alpha_star_healing = _get_alpha_max_healing(alpha_metrics)
            
            if alpha_IC == 0:
                recovery_IC = 0.0
            else:
                recovery_IC = (alpha_metrics[alpha_IC]['utility'] - baseline_metrics['mix']['utility'])/(baseline_metrics['clean']['utility'] - baseline_metrics['mix']['utility'])
            recovery_IC = _fmt_perct(recovery_IC)
            
            mix_metrics  = _fmt_metrics(baseline_metrics['mix'])
            clean_metrics = _fmt_metrics(baseline_metrics['clean'])
            rnd_metrics = _fmt_metrics(baseline_metrics['rnd'])
            
            if alpha_IC == 0:
                alpha_IC_metrics = mix_metrics
            else: alpha_IC_metrics = _fmt_metrics(alpha_metrics[alpha_IC])
            alpha_star_utility_metrics = _fmt_metrics(alpha_metrics[alpha_star_utility])
            alpha_star_healing_metrics = _fmt_metrics(alpha_metrics[alpha_star_healing])
            
            row_theta_mix[ds][j] = mix_metrics['utility']
            row_theta_clean[ds][j] = clean_metrics['utility']
            row_alpha_star_u[ds][j] = alpha_star_utility_metrics['utility']
            row_alpha_star_h[ds][j] = alpha_star_healing_metrics['utility']
            row_alpha_IC[ds][j] = alpha_IC_metrics['utility']
            row_random_vec[ds][j] = rnd_metrics['utility']
            row_recovery_IC[ds][j] = recovery_IC
    
            

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
        row_line(r"$\tau_{r}$", row_random_vec),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\alpha^\ast_a$", row_alpha_star_u),
        row_line(r"$\alpha^\ast_h$", row_alpha_star_h),
        r"\cmidrule(lr){1-16}",
        row_line(r"$\hat{\alpha}^\ast_{\text{IC}}$", row_alpha_IC),
        row_line(r"recovery", row_recovery_IC),
        
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer

    outputfile_path.write_text(table_tex)
    return outputfile_path



def generate_clip_IC_fr_dr_hr_table(
    results_dir:Path,
    cfgmap:OrderedDict,
    outputfile_path:Path = Path("./visulaization_dir/clip_IC_noise_fr_dr_hr_table.txt")  
):
    """
    Build a CIFAR-10 table showing, for each noise level, the metrics at three alpha choices:
      - alpha*_a (best utility)
      - alpha*_f (first alpha meeting forgetting threshold)
      - alpha_hat*_knn (kNN-estimated alpha)
    Rows: UT, RR, FR, HR, DR
    """
    
    corruption_classes = {
        "MNIST": ['1-7'],
        "CIFAR10": ['3-5', '6-9'],
        "CIFAR100": ['47-52', '0-24'],
    }

    ut_vals = ["-"] * len(corruption_classes['CIFAR10'])*3
    rr_vals = ["-"] * len(corruption_classes['CIFAR10'])*3
    fr_vals = ["-"] * len(corruption_classes['CIFAR10'])*3
    hr_vals = ["-"] * len(corruption_classes['CIFAR10'])*3
    dr_vals = ["-"] * len(corruption_classes['CIFAR10'])*3

    for i, class_maps in enumerate(corruption_classes['CIFAR10']):
        config = cfgmap.get(class_maps)
        if not config:
            continue

        metrics = _load_metrics(results_dir / config)
        if not metrics:
            continue

        # Clean up non-alpha keys that appear in files
        metrics.pop("FT HO Clean", None)
        alpha_IC = metrics.pop("alpha_IC", None)


        # Baselines and alpha grid
        baseline_metrics = _collect_baseline_metrics(metrics)  # pops Mix/Gold/RV out of `metrics`
        alpha_metrics = _collect_alpha_metrics(metrics)        # keys are rounded to 2 decimals

        # Resolve alphas
        alpha_a = _get_alpha_star_utility(alpha_metrics)  # best utility alpha
        alpha_h = _get_alpha_max_healing(alpha_metrics)

        alpha_list = [alpha_a, alpha_h, alpha_IC]

        # For each of the 3 alpha choices, fill 1 cell per row
        for j, alpha in enumerate(alpha_list):
            col_idx = i * 3 + j  # position within the 15 columns

            if alpha==0:
                a_metrics = baseline_metrics["mix"]
            else:
                a_metrics = alpha_metrics[alpha]
            ut = a_metrics.get("utility", None)
            fr = a_metrics.get("forget_rate", None)
            hr = a_metrics.get("healing_rate", None)
            dr = a_metrics.get("destruction_rate", None)

            # Recovery rate relative to mix→clean gap
            mix_ut = baseline_metrics["mix"]["utility"]
            clean_ut = baseline_metrics["clean"]["utility"]
            denom = (clean_ut - mix_ut) if (mix_ut is not None and clean_ut is not None) else None
            rr = 0.0
            if ut is not None and denom is not None and abs(denom) > 1e-12 and alpha_IC != 0:
                rr = (ut - mix_ut) / denom

            ut_vals[col_idx] = _fmt_perct(ut)
            rr_vals[col_idx] = _fmt_perct(rr)
            fr_vals[col_idx] = _fmt_perct(fr)
            hr_vals[col_idx] = _fmt_perct(hr)
            dr_vals[col_idx] = _fmt_perct(dr)

    # ---------- render LaTeX ----------
    def row_line(label: str, values: list[str]) -> str:
        return f"{label} & " + " & ".join(values) + r" \\"

    header = r"""\begin{table}[ht]
\centering
\label{tab:clip_sym_utility_vs_alpha}
\scriptsize
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccccccccc}
\toprule
& \multicolumn{15}{c}{CIFAR10} \\
\cmidrule(lr){2-16}
& \multicolumn{3}{c}{10\%} & \multicolumn{3}{c}{20\%} & \multicolumn{3}{c}{40\%} & \multicolumn{3}{c}{60\%}  & \multicolumn{3}{c}{80\%}  \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}
Metric & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ & $\alpha^\ast_a$ & $\alpha^\ast_f$ & $\hat{\alpha}^\ast_{\text{knn}}$ \\
\midrule
"""
    body_lines = [
        row_line("UT", ut_vals),
        row_line("RR", rr_vals),
        row_line("FR", fr_vals),
        row_line("HR", hr_vals),
        row_line("DR", dr_vals),
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer
    outputfile_path.write_text(table_tex)
    return outputfile_path



def generate_clip_poison_table(
    results_dir: Path,
    cfgmap: OrderedDict,
    dataset_order: List[str] = ["MNIST", "CIFAR10", "CIFAR100"],
    dataset_forget_trsh: Dict[str, float] = {
        "MNIST": 0.99,
        "CIFAR10": 0.99,
        "CIFAR100": 0.99,
    },
    outputfile_path: Path = Path("./visulaization_dir/clip_poison_triggers_table.txt"),
) -> Path:
    """
    Build a single table (MNIST, CIFAR-10, CIFAR-100) for poison-trigger experiments.
    Columns for each dataset: UT, FR, HR, DR.
    Rows: θ_mix, θ_clean, τ_r, α*_a, α*_f, α̂*_psn.

    Assumptions:
      - cfgmap maps dataset name -> relative config path (no η levels).
      - metrics.json contains 'Mix', 'Gold', 'Random Vector', float-α entries, and 'alpha_psn'.
      - If alpha_psn == 0, we report Mix metrics for that row.
    """

    # Per-row containers: 4 cells per dataset
    row_theta_mix   = {ds: ["-"] * 4 for ds in dataset_order}
    row_theta_clean = {ds: ["-"] * 4 for ds in dataset_order}
    row_tau_r       = {ds: ["-"] * 4 for ds in dataset_order}
    row_alpha_a     = {ds: ["-"] * 4 for ds in dataset_order}
    row_alpha_f     = {ds: ["-"] * 4 for ds in dataset_order}
    row_alpha_psn   = {ds: ["-"] * 4 for ds in dataset_order}

    def _cells_fmt(m: Dict[str, Optional[float]]) -> List[str]:
        fm = _fmt_metrics(m)
        # order: UT, FR, HR, DR
        return [
            fm.get("utility", "-"),
            fm.get("forget_rate", "-"),
            fm.get("healing_rate", "-"),
            fm.get("destruction_rate", "-"),
        ]

    for ds in dataset_order:
        config_rel = cfgmap.get(ds)
        if not config_rel:
            continue

        metrics = _load_metrics(results_dir / config_rel)
        if not metrics:
            continue

        # Clean out non-alpha extras; pull alpha_psn
        metrics.pop("FT HO Clean", None)
        metrics.pop("alpha_s4", None)
        alpha_psn = float(metrics.pop("alpha_psn", None))


        # Baselines (pops Mix/Gold/RV from metrics)
        baseline = _collect_baseline_metrics(metrics)
        # Alpha grid
        alpha_grid = _collect_alpha_metrics(metrics)

        # Fill baseline rows
        row_theta_mix[ds]   = _cells_fmt(baseline["mix"])
        row_theta_clean[ds] = _cells_fmt(baseline["clean"])
        row_tau_r[ds]       = _cells_fmt(baseline["rnd"])

        # Resolve alpha*_a and alpha*_f
        alpha_a = _get_alpha_star_utility(alpha_grid)
        alpha_f = _get_alpha_star_forgetting(alpha_grid, dataset_forget_trsh.get(ds, 0.9))

        if alpha_a in alpha_grid:
            row_alpha_a[ds] = _cells_fmt(alpha_grid[alpha_a])
        if alpha_f in alpha_grid:
            row_alpha_f[ds] = _cells_fmt(alpha_grid[alpha_f])

        # Resolve alpha_psn → row_alpha_psn
        psn_cells = ["-"] * 4
        if alpha_psn is not None:
            psn_cells = _cells_fmt(alpha_grid[alpha_psn])

        row_alpha_psn[ds] = psn_cells

    # ---------- render LaTeX ----------
    def row_line(label: str, rows_by_ds: Dict[str, List[str]]) -> str:
        cells: List[str] = []
        for ds in dataset_order:
            cells.extend(rows_by_ds[ds])
        return f"{label} & " + " & ".join(cells) + r" \\"

    header = r"""\begin{table}[ht]
\centering
\caption{}
\label{tab:clip_sym_utility_vs_alpha}
\scriptsize
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccccccccc}
\toprule
& \multicolumn{4}{c}{MNIST} & \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{CIFAR-100} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} 
Model & UT $\uparrow$  & FR $\uparrow$ & HR $\uparrow$ & DR $\downarrow$ & UT $\uparrow$  & FR $\uparrow$ & HR $\uparrow$ & DR $\downarrow$ & UT $\uparrow$  & FR $\uparrow$ & HR $\uparrow$ & DR $\downarrow$\\
\midrule
"""

    body_lines = [
        row_line(r"$\theta_{\text{mix}}$",   row_theta_mix),
        row_line(r"$\theta_{\text{clean}}$", row_theta_clean),
        r"\cmidrule(lr){1-13}",
        row_line(r"$\tau_{r}$",              row_tau_r),
        r"\cmidrule(lr){1-13}",
        row_line(r"$\alpha^\ast_a$",         row_alpha_a),
        row_line(r"$\alpha^\ast_f$",         row_alpha_f),
        r"\cmidrule(lr){1-13}",
        row_line(r"$\hat{\alpha}^\ast_{\text{psn}}$", row_alpha_psn),
    ]

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_tex = header + "\n".join(body_lines) + footer
    outputfile_path.parent.mkdir(parents=True, exist_ok=True)
    outputfile_path.write_text(table_tex)
    return outputfile_path


def plot_alpha_poison_interplay_dual(
    results_dir: Path,
    config_rel_path_A: str,
    config_rel_path_B: str,
    dataset_name_A: str = "CIFAR10",
    dataset_name_B: str = "CIFAR10",
    forget_threshold_A: float = 0.9,
    forget_threshold_B: float = 0.9,
    save_path: Path = Path("./visulaization_dir/anonymos_poison_triggers_plot.png"),
) -> Path:
    """
    Make a 1x2 figure of UT/FR/HR/DR vs α for two experiments (A, B), with:
      - Preferred x-ordering: only nonpositive alphas (<=0), 0 at left, more negative to the right.
      - Fallback: if no negatives exist, plot against |alpha| ascending.
      - Mix injected at α=0.
      - Clean UT horizontal reference line.
      - α*_a, α*_f, α_kNN highlighted on x-axis (bottom labels).
      - Shared y-axis and a single, shared legend.

    Saves to: ./visulaization_dir/interplay_{nameA}__{nameB}.png
    Returns: saved Path
    """

    def _prepare_one(
        config_rel_path: str,
        dataset_name: str,
        forget_threshold: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load metrics, inject mix@0, compute alphas, choose ordering, return plotting payload + meta."""
        metrics = _load_metrics(results_dir / config_rel_path)
        if not metrics:
            raise FileNotFoundError(f"metrics.json not found or unreadable for {config_rel_path}")

        # clean out non-alpha entries; extract alpha_psn
        metrics.pop("FT HO Clean", None)
        metrics.pop("alpha_s4", None)
        alpha_psn = float(metrics.pop("alpha_psn", None))
        try:
            alpha_psn = None if alpha_psn is None else round(float(alpha_psn), 2)
        except Exception:
            alpha_psn = None

        baselines = _collect_baseline_metrics(metrics)   # pops Mix/Gold/Random Vector
        alpha_grid = _collect_alpha_metrics(metrics)     # keys are rounded floats

        # Inject Mix at α=0
        mix_alpha = 0.0
        alpha_grid[mix_alpha] = {
            "utility": baselines["mix"].get("utility", 0),
            "forget_rate": baselines["mix"].get("forget_rate", 0),
            "healing_rate": baselines["mix"].get("healing_rate", 0),
            "destruction_rate": baselines["mix"].get("destruction_rate", 0),
        }

        # Resolve α*’s
        alpha_star_u = _get_alpha_star_utility(alpha_grid)
        alpha_star_f = _get_alpha_star_forgetting(alpha_grid, forget_threshold)

        # Choose x mapping & ordering
        all_as = list(alpha_grid.keys())
        neg_as = sorted([a for a in all_as if a < 0], key=lambda a: abs(a))  # -0.01, -0.02, ...
        use_abs_mode = (len(neg_as) == 0)

        if not use_abs_mode:
            ordered_alphas: List[float] = [mix_alpha] + neg_as
            x_vals = ordered_alphas
            xlabel = r"$\alpha$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return a if (a is not None and a in alpha_grid and a <= 0.0) else None
            x_left, x_right = 0.0, (min(neg_as) if neg_as else 0.0)  # invert axis
        else:
            ordered_alphas = sorted(all_as, key=lambda a: (abs(a), a))
            x_vals = [abs(a) for a in ordered_alphas]
            xlabel = r"$|\alpha|$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return abs(a) if (a is not None and a in alpha_grid) else None
            x_left, x_right = 0.0, (max(x_vals) if x_vals else 0.0)

        payload = dict(
            dataset_name=dataset_name,
            config_rel_path=config_rel_path,
            baselines=baselines,
            alpha_grid=alpha_grid,
            alpha_star_u=alpha_star_u,
            alpha_star_f=alpha_star_f,
            alpha_psn=alpha_psn,
            x_vals=x_vals,
            xlabel=xlabel,
            xlim=(x_left, x_right),
            UT=_series("utility"),
            FR=_series("forget_rate"),
            HR=_series("healing_rate"),
            DR=_series("destruction_rate"),
            ann_x=_ann_x,
            use_abs_mode=use_abs_mode,
        )
        meta = dict()  # reserved for future needs
        return payload, meta

    # Prepare both panels
    payloadA, _ = _prepare_one(config_rel_path_A, dataset_name_A, forget_threshold_A)
    payloadB, _ = _prepare_one(config_rel_path_B, dataset_name_B, forget_threshold_B)

    # ---------- plotting ----------

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=220, sharey=True)

    def _plot_panel(ax, P: Dict[str, Any], title: Optional[str] = None, add_legend: bool = False):
        # Lines only (no markers)
        h_ut, = ax.plot(P["x_vals"], P["UT"], linewidth=2.2, label="UT")
        h_fr, = ax.plot(P["x_vals"], P["FR"], linewidth=2.0, label="FR")
        h_hr, = ax.plot(P["x_vals"], P["HR"], linewidth=2.0, label="HR")
        h_dr, = ax.plot(P["x_vals"], P["DR"], linewidth=2.0, label="DR")

        # Axes styling
        ax.set_xlabel(P["xlabel"], fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(*P["xlim"])

        # Panel title
        if title is None:
            title = f"{P['dataset_name']} · {Path(P['config_rel_path']).parent.name}"
        ax.set_title(title, fontsize=12)

        # Clean UT reference
        clean_ut = P["baselines"]["clean"]["utility"]
        if clean_ut is not None:
            ax.axhline(clean_ut, color="#BBBBBB", linewidth=1.5, linestyle="--", alpha=0.9)

        # Alpha annotations on x-axis (bottom)
        def _annotate_on_axis(x_at: Optional[float], label: str, linestyle, y_margin=0.015):
            if x_at is None:
                return
            ymin, ymax = ax.get_ylim()
            ax.axvline(x_at, linestyle=linestyle, linewidth=1.1, alpha=0.6, color="black")
            ax.text(
                x_at,
                ymin + y_margin * (ymax - ymin),
                label,
                va="bottom",
                ha="center",
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
                clip_on=False,
            )

        _annotate_on_axis(P["ann_x"](P["alpha_star_u"]), r"$\alpha^\ast_a$", linestyle=":")
        _annotate_on_axis(P["ann_x"](P["alpha_star_f"]), r"$\alpha^\ast_f$", linestyle="--", y_margin=0.08)
        _annotate_on_axis(P["ann_x"](P["alpha_psn"]),   r"$\hat{\alpha}^\ast_{\mathrm{psn}}$", linestyle="-.")

        # return handles for a single shared legend if requested
        if add_legend:
            return [h_ut, h_fr, h_hr, h_dr]
        return []

    # Left panel: add legend handles
    handles = _plot_panel(axes[0], payloadA, title=f"{payloadA['dataset_name']}", add_legend=True)
    # Right panel
    _plot_panel(axes[1], payloadB, title=f"{payloadB['dataset_name']}")

    # Shared Y label on the figure
    fig.text(0.04, 0.5, "Metric value", va="center", rotation="vertical", fontsize=11)

    # Single shared legend (deduplicated by dict)
    by_label = {}
    for h in handles:
        by_label[h.get_label()] = h
    fig.legend(list(by_label.values()), list(by_label.keys()),
               loc="lower center", ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.52, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(left=0.09, bottom=0.18)  # increase left from default

    fig.subplots_adjust(bottom=0.18)  # make room for shared legend
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return save_path


def plot_alpha_noise_and_poison_interplay_dual(
    results_dir_noise: Path,
    config_rel_path_noise: str,
    results_dir_poison: Path,
    config_rel_path_poison: str,
    dataset_name_noise: str = "CIFAR10",
    dataset_name_poison: str = "CIFAR10",
    forget_threshold_noise: float = 0.9,
    forget_threshold_poison: float = 0.9,
    save_path: Path = Path("./visulaization_dir/noise_and_poison_interplay_dual.png"),
) -> Path:
    """
    Make a 1x2 figure of UT/FR/HR/DR vs α for two experiments of DIFFERENT types:
      - Left (A): NOISE experiment (expects 'alpha_KNN' in metrics.json) → annotate kNN alpha
      - Right (B): POISON experiment (expects 'alpha_psn' in metrics.json) → annotate poison alpha

    Shared behavior:
      - Preferred x-ordering: alphas <= 0 only, with 0 at left and more negative to the right.
      - Fallback: if no negatives exist, plot against |alpha| ascending.
      - Mix injected at α=0.
      - Clean UT horizontal reference line.
      - α*_a, α*_f, and alpha_key highlighted on x-axis (bottom labels).
      - Shared y-axis and a single, shared legend.
    Saves to: ./visulaization_dir/interplay_noise_vs_poison_{nameA}__{nameB}.png
    Returns: saved Path
    """

    def _prepare_one(
        results_dir: Path,
        config_rel_path: str,
        dataset_name: str,
        forget_threshold: float,
        alpha_key: str,  # 'alpha_KNN' OR 'alpha_psn'
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load metrics, inject mix@0, compute alphas, choose ordering, return plotting payload + meta."""
        metrics = _load_metrics(results_dir / config_rel_path)
        if not metrics:
            raise FileNotFoundError(f"metrics.json not found or unreadable for {config_rel_path}")

        # Clean out non-alpha extras; extract alpha_key
        metrics.pop("FT HO Clean", None)
        metrics.pop("alpha_s4", None)

        alpha_raw = metrics.pop(alpha_key, None)
        try:
            alpha_special = None if alpha_raw is None else round(float(alpha_raw), 2)
        except Exception:
            alpha_special = None

        baselines = _collect_baseline_metrics(metrics)   # pops Mix/Gold/Random Vector
        alpha_grid = _collect_alpha_metrics(metrics)     # keys are rounded floats

        # Inject Mix at α=0
        mix_alpha = 0.0
        alpha_grid[mix_alpha] = {
            "utility": baselines["mix"].get("utility", 0),
            "forget_rate": baselines["mix"].get("forget_rate", 0),
            "healing_rate": baselines["mix"].get("healing_rate", 0),
            "destruction_rate": baselines["mix"].get("destruction_rate", 0),
        }

        # Resolve α*’s
        alpha_star_u = _get_alpha_star_utility(alpha_grid)
        alpha_star_f = _get_alpha_star_forgetting(alpha_grid, forget_threshold)

        # Choose x mapping & ordering
        all_as = list(alpha_grid.keys())
        neg_as = sorted([a for a in all_as if a < 0], key=lambda a: abs(a))  # -0.01, -0.02, ...
        use_abs_mode = (len(neg_as) == 0)

        if not use_abs_mode:
            ordered_alphas: List[float] = [mix_alpha] + neg_as
            x_vals = ordered_alphas
            xlabel = r"$\alpha$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return a if (a is not None and a in alpha_grid and a <= 0.0) else None
            x_left, x_right = 0.0, (min(neg_as) if neg_as else 0.0)  # invert axis
        else:
            ordered_alphas = sorted(all_as, key=lambda a: (abs(a), a))
            x_vals = [abs(a) for a in ordered_alphas]
            xlabel = r"$|\alpha|$"
            def _series(k: str): return [alpha_grid[a].get(k, None) for a in ordered_alphas]
            def _ann_x(a: Optional[float]) -> Optional[float]:
                return abs(a) if (a is not None and a in alpha_grid) else None
            x_left, x_right = 0.0, (max(x_vals) if x_vals else 0.0)

        payload = dict(
            dataset_name=dataset_name,
            config_rel_path=config_rel_path,
            baselines=baselines,
            alpha_grid=alpha_grid,
            alpha_star_u=alpha_star_u,
            alpha_star_f=alpha_star_f,
            alpha_special=alpha_special,   # kNN or psn depending on panel
            x_vals=x_vals,
            xlabel=xlabel,
            xlim=(x_left, x_right),
            UT=_series("utility"),
            FR=_series("forget_rate"),
            HR=_series("healing_rate"),
            DR=_series("destruction_rate"),
            ann_x=_ann_x,
            use_abs_mode=use_abs_mode,
        )
        meta = dict(alpha_key=alpha_key)
        return payload, meta

    # Prepare A (noise, alpha_KNN) and B (poison, alpha_psn)
    payloadA, metaA = _prepare_one(
        results_dir_noise, config_rel_path_noise, dataset_name_noise, forget_threshold_noise, alpha_key="alpha_KNN"
    )
    payloadB, metaB = _prepare_one(
        results_dir_poison, config_rel_path_poison, dataset_name_poison, forget_threshold_poison, alpha_key="alpha_psn"
    )

    # ---------- plotting ----------


    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=220, sharey=True)

    def _plot_panel(ax, P: Dict[str, Any], title: Optional[str], special_label: str, add_legend: bool = False):
        # Lines only (no markers)
        h_ut, = ax.plot(P["x_vals"], P["UT"], linewidth=2.2, label="UT")
        h_fr, = ax.plot(P["x_vals"], P["FR"], linewidth=2.0, label="FR")
        h_hr, = ax.plot(P["x_vals"], P["HR"], linewidth=2.0, label="HR")
        h_dr, = ax.plot(P["x_vals"], P["DR"], linewidth=2.0, label="DR")

        # Axes styling
        ax.set_xlabel(P["xlabel"], fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(*P["xlim"])

        # Panel title
        if title is None:
            title = f"{P['dataset_name']} · {Path(P['config_rel_path']).parent.name}"
        ax.set_title(title, fontsize=12)

        # Clean UT reference
        clean_ut = P["baselines"]["clean"]["utility"]
        if clean_ut is not None:
            ax.axhline(clean_ut, color="#BBBBBB", linewidth=1.5, linestyle="--", alpha=0.9)

        # Alpha annotations on x-axis (bottom)
        def _annotate_on_axis(x_at: Optional[float], label: str, linestyle, y_margin=0.015):
            if x_at is None:
                return
            ymin, ymax = ax.get_ylim()
            ax.axvline(x_at, linestyle=linestyle, linewidth=1.1, alpha=0.6, color="black")
            ax.text(
                x_at,
                ymin + y_margin * (ymax - ymin),
                label,
                va="bottom",
                ha="center",
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
                clip_on=False,
            )

        _annotate_on_axis(P["ann_x"](P["alpha_star_u"]), r"$\alpha^\ast_a$", linestyle=":")
        _annotate_on_axis(P["ann_x"](P["alpha_star_f"]), r"$\alpha^\ast_f$", linestyle="--", y_margin=0.08)
        _annotate_on_axis(P["ann_x"](P["alpha_special"]), special_label, linestyle="-.")

        # return handles for a single shared legend if requested
        if add_legend:
            return [h_ut, h_fr, h_hr, h_dr]
        return []

    # Left panel (NOISE): annotate kNN alpha
    handles = _plot_panel(
        axes[0],
        payloadA,
        title=f"{payloadA['dataset_name']}",
        special_label=r"$\hat{\alpha}^\ast_{\mathrm{kNN}}$",
        add_legend=True,
    )
    # Right panel (POISON): annotate poison alpha
    _plot_panel(
        axes[1],
        payloadB,
        title=f"{payloadB['dataset_name']}",
        special_label=r"$\hat{\alpha}^\ast_{\mathrm{psn}}$",
        add_legend=False,
    )

    # Shared Y label on the figure
    fig.text(0.04, 0.5, "Metric value", va="center", rotation="vertical", fontsize=11)

    # Single shared legend (deduplicated by dict)
    by_label = {}
    for h in handles:
        by_label[h.get_label()] = h
    fig.legend(
        list(by_label.values()),
        list(by_label.keys()),
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.52, -0.02),
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.09, bottom=0.18)  # space for y-label & legend
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return save_path






if __name__ == "__main__":
    with open('configmap.yaml', 'r') as file:
        configmap = yaml.full_load(file)
    
    # ------------------- CLIP -------------------
    
    clip_models_cfgs = configmap['clip_models']
    clip_noise_results_dir = Path('results/single_experiment/clip_noise_TA')
    clip_ic_results_dir = Path('results/single_experiment/clip_IC_TA')
    clip_poison_results_dir = Path('results/single_experiment/clip_poison_TA')
    
    
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
    
    # ------------------- DINO -------------------
    dino_models_cfgs = configmap['DINO']
    dino_noise_results_dir = Path('results/single_experiment/dino_noise_TA')
    dino_poison_results_dir = Path('results/single_experiment/dino_poison_TA')
    
    dino_symmetric_cfgs = OrderedDict()
    dino_symmetric_cfgs['MNIST'] = dino_models_cfgs['MNIST']['ho_2']['symmetric']
    dino_symmetric_cfgs['CIFAR10'] = dino_models_cfgs['CIFAR10']['ho_2']['symmetric']
    dino_symmetric_cfgs['CIFAR100'] = dino_models_cfgs['CIFAR100']['ho_2']['symmetric']
    
    dino_asymmetric_cfgs = OrderedDict()
    dino_asymmetric_cfgs['CIFAR10'] = dino_models_cfgs['CIFAR10']['ho_2']['asymmetric']
    dino_asymmetric_cfgs['CIFAR100'] = dino_models_cfgs['CIFAR100']['ho_2']['asymmetric']
    
    dino_poison_cfgs = OrderedDict()
    dino_poison_cfgs['CIFAR10'] = dino_models_cfgs['CIFAR10']['ho_2']['poison']
    dino_poison_cfgs['CIFAR100'] = dino_models_cfgs['CIFAR100']['ho_2']['poison']
    
    
    # ------------------- regular models -------------------
    regular_models_cfgs = configmap['regular_models']
    regular_noise_results_dir = Path('results/single_experiment/regular_noise_TA')
    regular_poison_results_dir = Path('results/single_experiment/regular_poison_TA')
    
    regular_symmetric_cfgs = OrderedDict()
    regular_symmetric_cfgs['MNIST'] = regular_models_cfgs['MNIST']['scratch']['ho_2']['symmetric']['fc1']
    regular_symmetric_cfgs['CIFAR10'] = regular_models_cfgs['CIFAR10']['scratch']['ho_2']['symmetric']['resnet18']
    regular_symmetric_cfgs['CIFAR100'] = regular_models_cfgs['CIFAR100']['scratch']['ho_2']['symmetric']['resnet18']
    
    
    regular_asymmetric_cfgs = OrderedDict()
    regular_asymmetric_cfgs['MNIST'] = regular_models_cfgs['MNIST']['scratch']['ho_2']['asymmetric']['fc1']
    regular_asymmetric_cfgs['CIFAR10'] = regular_models_cfgs['CIFAR10']['scratch']['ho_2']['asymmetric']['resnet18']
    regular_asymmetric_cfgs['CIFAR100'] = regular_models_cfgs['CIFAR100']['scratch']['ho_2']['asymmetric']['resnet18']
    
    
    regular_poison_cfgs = OrderedDict()
    regular_poison_cfgs['MNIST'] = regular_models_cfgs['MNIST']['scratch']['ho_2']['poison']['fc1']
    regular_poison_cfgs['CIFAR10'] = regular_models_cfgs['CIFAR10']['scratch']['ho_2']['poison']['resnet18']
    regular_poison_cfgs['CIFAR100'] = regular_models_cfgs['CIFAR100']['scratch']['ho_2']['poison']['resnet18']
    
    
    regular_symmetric_comp_cfgs = OrderedDict()
    regular_symmetric_comp_cfgs['CIFAR10'] = OrderedDict({
        'resnet18': (regular_models_cfgs['CIFAR10']['scratch']['ho_2']['symmetric']['resnet18'], regular_models_cfgs['CIFAR10']['pretrained']['ho_2']['symmetric']['resnet18']),
        'resnet34': (regular_models_cfgs['CIFAR10']['scratch']['ho_2']['symmetric']['resnet34'], regular_models_cfgs['CIFAR10']['pretrained']['ho_2']['symmetric']['resnet34']),
        'resnet50': (regular_models_cfgs['CIFAR10']['scratch']['ho_2']['symmetric']['resnet50'], regular_models_cfgs['CIFAR10']['pretrained']['ho_2']['symmetric']['resnet50']),
        'resnet101': (regular_models_cfgs['CIFAR10']['scratch']['ho_2']['symmetric']['resnet101'], regular_models_cfgs['CIFAR10']['pretrained']['ho_2']['symmetric']['resnet101'])
    })

    
    
    #################################################################################
    #########                           CLIP models                         #########
    
    # generate_clip_noise_utlity_table(clip_noise_results_dir, clip_symmetric_cfgs)
    # generate_clip_symmetric_noise_fr_dr_hr_table(clip_noise_results_dir, clip_symmetric_cfgs['CIFAR10'])
    # plot_alpha_interplay(clip_noise_results_dir, clip_symmetric_cfgs['CIFAR10'][60])
    # plot_alpha_interplay_dual(
    #     clip_noise_results_dir,
    #     clip_symmetric_cfgs['CIFAR10'][60],
    #     clip_symmetric_cfgs['CIFAR100'][10],
    #     dataset_name_A="CIFAR-10 (60%)",
    #     dataset_name_B="CIFAR-100 (10%)",
    #     forget_threshold_A=0.9,
    #     forget_threshold_B=0.9,
    # )
    
    
    # generate_clip_noise_utlity_table(
    #     clip_noise_results_dir,
    #     clip_asymmetric_cfgs,
    #     dataset_order=['CIFAR10', 'CIFAR100'],
    #     dataset_forget_trsh={
    #         'MNIST': 0.9,
    #         'CIFAR10': 0.89,
    #         'CIFAR100': 0.89
    #     },
    #     noise_levels=[20, 40],
    #     outputfile_path=Path('visulaization_dir/clip_asymmetric_noise_table.txt')
    #     )
    
    # generate_clip_noise_fr_dr_hr_table(
    #     clip_noise_results_dir,
    #     clip_symmetric_cfgs['CIFAR10'],
    #     noise_levels=[20, 40]
    #     )
    
    # plot_alpha_interplay_dual(
    #     clip_noise_results_dir,
    #     clip_asymmetric_cfgs['CIFAR10'][40],
    #     clip_asymmetric_cfgs['CIFAR100'][20],
    #     dataset_name_A="CIFAR-10 (40%)",
    #     dataset_name_B="CIFAR-100 (20%)",
    #     forget_threshold_A=0.89,
    #     forget_threshold_B=0.89,
    # )
    
    # generate_clip_IC_utlity_table(clip_ic_results_dir, clip_ic_cfgs)
    # generate_clip_IC_fr_dr_hr_table(clip_ic_results_dir, clip_ic_cfgs['CIFAR10'])
    # generate_clip_poison_table(
    #     clip_poison_results_dir,
    #     clip_poison_cfgs,
    #     outputfile_path=Path("./visulaization_dir/clip_poison_triggers_table.txt")
    #     )
    # plot_alpha_poison_interplay_dual(
    #     clip_poison_results_dir,
    #     clip_poison_cfgs['CIFAR10'],
    #     clip_poison_cfgs['CIFAR100'],
    #     dataset_name_A="CIFAR-10 (2%)",
    #     dataset_name_B="CIFAR-100 (2%)",
    #     forget_threshold_A=0.99,
    #     forget_threshold_B=0.99,
    #     save_path=Path("./visulaization_dir/clip_poison_triggers_plot.png")
    # )
    
    
    
    #################################################################################
    #########                           DINO models                         #########
    
    # generate_clip_noise_utlity_table(
    #     dino_noise_results_dir,
    #     dino_asymmetric_cfgs,
    #     dataset_order=['CIFAR10', 'CIFAR100'],
    #     noise_levels=[40],
    #     outputfile_path=Path("./visulaization_dir/dino_asymmetric_noise_table.txt")
    # )
    # generate_clip_poison_table(
    #     dino_poison_results_dir,
    #     dino_poison_cfgs,
    #     dataset_order= ["CIFAR10", "CIFAR100"],
    #     outputfile_path=Path("./visulaization_dir/dino_poison_trigger_table.txt")
    # )
    
    
    plot_alpha_noise_and_poison_interplay_dual(
        results_dir_noise=dino_noise_results_dir,
        config_rel_path_noise=dino_symmetric_cfgs['CIFAR10'][40],  # NOISE exp
        results_dir_poison=dino_poison_results_dir,
        config_rel_path_poison=dino_poison_cfgs['CIFAR10'],        # POISON exp
        dataset_name_noise="CIFAR-10 (40%)",
        dataset_name_poison="CIFAR-10 (2%)",
        forget_threshold_noise=0.89,
        forget_threshold_poison=0.99,
        save_path=Path("./visulaization_dir/dino_noise_poison_interplay_plot.png"),
    )
    
    
    
    #################################################################################
    #########                        Regular models                         #########
    
    # generate_clip_noise_utlity_table(
    #     regular_noise_results_dir,
    #     regular_symmetric_cfgs,
    #     outputfile_path= Path("./visulaization_dir/regular_symmetric_noise_table.txt")
    #     )

    