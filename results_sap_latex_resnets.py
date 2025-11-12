#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional
import yaml

CONFIGMAP_PATH = Path("configmap.yaml")
BASE_DIR = Path("results/single_experiment/regular_noise_TA")

# Column order within each block
SN_LEVELS = [10, 20, 40, 60, 80]
AN_LEVELS = [40]  # only 40% for AN in your map

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.full_load(f)

def best_test_acc_from_metrics(json_path: Path) -> Optional[float]:
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        best_alpha = data.get("Best Alpha")
        if best_alpha is None:
            return None
        entry = data.get(str(best_alpha))
        if not entry:
            return None
        acc = entry.get("Test", {}).get("ACC")
        return float(acc) if isinstance(acc, (int, float)) else None
    except Exception:
        return None

def get_cfg(regular_models: dict, dataset: str, arch: str, kind: str, level: int) -> Optional[str]:
    """
    Traverse:
    regular_models[dataset]['scratch']['ho_2'][kind][arch][level] -> 'configXX'
    """
    try:
        node = (
            regular_models[dataset]
            .get("scratch", {})
            .get("ho_2", {})
            .get(kind, {})
            .get(arch, {})
        )
        return node.get(level)
    except Exception:
        return None

def pull_values(regular_models: dict, dataset: str, arch: str):
    # SN (10,20,40,60,80)
    vals = []
    for lvl in SN_LEVELS:
        cfg = get_cfg(regular_models, dataset, arch, "symmetric", lvl)
        acc = best_test_acc_from_metrics(BASE_DIR / cfg / "metrics_sap.json") if cfg else None
        vals.append(acc)
    # AN (40)
    for lvl in AN_LEVELS:
        cfg = get_cfg(regular_models, dataset, arch, "asymmetric", lvl)
        acc = best_test_acc_from_metrics(BASE_DIR / cfg / "metrics_sap.json") if cfg else None
        vals.append(acc)
    return vals  # length 6

def fmt(x: Optional[float]) -> str:
    return f"{100.0 * round(x, 3):.1f}" if isinstance(x, (int, float)) else "--"

def main():
    cfg = load_yaml(CONFIGMAP_PATH)
    regular_models = cfg["regular_models"]

    # Build groups in table order
    mnist_fc1 = pull_values(regular_models, "MNIST", "fc1")
    c10_r18   = pull_values(regular_models, "CIFAR10", "resnet18")
    c100_r18  = pull_values(regular_models, "CIFAR100", "resnet18")

    # LaTeX row: & MNIST(6) & & C10(6) & & C100(6) \\
    row_parts = (
        ["&"]
        + [fmt(v) for v in mnist_fc1]
        + ["&", "&"]
        + [fmt(v) for v in c10_r18]
        + ["&", "&"]
        + [fmt(v) for v in c100_r18]
    )
    latex_row = " ".join(row_parts) + r" \\"
    print(latex_row)

    # Optional labeled debug
    labels = (
        [f"MNIST+FC1 SN {p}%" for p in SN_LEVELS] + ["MNIST+FC1 AN 40%"] +
        [f"C10+R18 SN {p}%" for p in SN_LEVELS] + ["C10+R18 AN 40%"] +
        [f"C100+R18 SN {p}%" for p in SN_LEVELS] + ["C100+R18 AN 40%"]
    )
    values = mnist_fc1 + c10_r18 + c100_r18
    for lab, val in zip(labels, values):
        print(f"{lab:>18}: {fmt(val)}")

if __name__ == "__main__":
    main()
