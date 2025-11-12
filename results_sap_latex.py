#!/usr/bin/env python3
import json
from pathlib import Path
from collections import OrderedDict
import yaml

BASE_DIR = Path("results/single_experiment/clip_noise_TA")
CONFIGMAP_PATH = Path("configmap.yaml")

# ---- Column order (19 total) ----
# MNIST SN: 10,20,40,60,80  (5)
MNIST_SN = [10, 20, 40, 60, 80]
# CIFAR10 SN: 10,20,40,60,80 (5) + AN: 20,40 (2)  -> 7 total
CIFAR10_SN = [10, 20, 40, 60, 80]
CIFAR10_AN = [20, 40]
# CIFAR100 SN: 10,20,40,60,80 (5) + AN: 20,40 (2) -> 7 total
CIFAR100_SN = [10, 20, 40, 60, 80]
CIFAR100_AN = [20, 40]

def load_yaml_configmap(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.full_load(f)

def best_test_acc_from_metrics(json_path: Path):
    """
    Returns float Test ACC at Best Alpha, or None if unavailable.
    """
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        best_alpha = data.get("Best Alpha", None)
        if best_alpha is None:
            return None
        # JSON keys for alphas are strings; ensure we index by string
        key = str(best_alpha)
        entry = data.get(key, None)
        if not entry:
            return None
        test_acc = entry.get("Test", {}).get("ACC", None)
        if isinstance(test_acc, (int, float)):
            return float(test_acc)
        return None
    except Exception:
        return None

def resolve_config(cfg_block: dict, level) -> str | None:
    """
    cfg_block is a dict mapping noise_level -> 'configXX'
    level is an int noise percentage
    """
    if not isinstance(cfg_block, dict):
        return None
    # YAML might have ints as ints; ensure we look up by int
    return cfg_block.get(level, None)

def pull_values_for_section(models_cfgs: dict, dataset: str, kind: str, levels: list[int]) -> list:
    """
    kind: 'symmetric' or 'asymmetric'
    """
    out = []
    ds_node = models_cfgs.get(dataset, {})
    ho2_node = ds_node.get("ho_2", {})
    kind_node = ho2_node.get(kind, {})
    for lvl in levels:
        cfg = resolve_config(kind_node, lvl)
        if cfg is None:
            out.append(None)
            continue
        json_path = BASE_DIR / cfg / "metrics_sap.json"
        acc = best_test_acc_from_metrics(json_path)
        out.append(acc)
    return out

def main():
    configmap = load_yaml_configmap(CONFIGMAP_PATH)
    clip_models_cfgs = configmap["clip_models"]

    # Build the 19 values in the exact table order:
    row_values = []

    # MNIST (SN: 10,20,40,60,80)
    row_values += pull_values_for_section(clip_models_cfgs, "MNIST", "symmetric", MNIST_SN)

    # CIFAR10 (SN) then (AN)
    row_values += pull_values_for_section(clip_models_cfgs, "CIFAR10", "symmetric", CIFAR10_SN)
    row_values += pull_values_for_section(clip_models_cfgs, "CIFAR10", "asymmetric", CIFAR10_AN)

    # CIFAR100 (SN) then (AN)
    row_values += pull_values_for_section(clip_models_cfgs, "CIFAR100", "symmetric", CIFAR100_SN)
    row_values += pull_values_for_section(clip_models_cfgs, "CIFAR100", "asymmetric", CIFAR100_AN)

    assert len(row_values) == 19, f"Expected 19 values, got {len(row_values)}"

    # Format as LaTeX row, numbers with 3 decimals; missing as --
    def fmt(x):
        return f"{100.0 * round(x, 3):.1f}" if isinstance(x, (int, float)) else "--"

    latex_row = " & " + " & ".join(fmt(v) for v in row_values) + r" \\"
    print(latex_row)

    # (Optional) also print a labeled view to verify alignment quickly
    labels = (
        [f"MNIST SN {p}%" for p in MNIST_SN]
        + [f"C10 SN {p}%" for p in CIFAR10_SN]
        + [f"C10 AN {p}%" for p in CIFAR10_AN]
        + [f"C100 SN {p}%" for p in CIFAR100_SN]
        + [f"C100 AN {p}%" for p in CIFAR100_AN]
    )
    for lab, val in zip(labels, row_values):
        print(f"{lab:>12}: {fmt(val)}")

if __name__ == "__main__":
    main()
