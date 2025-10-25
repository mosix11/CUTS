from pathlib import Path
import json
import matplotlib.pyplot as plt

def plot_acc_curves_from_jsons(json_paths, labels, metric_key="ACC", savepath=None, show=True):
    """
    - No markers (simple lines)
    - Colors: Test=blue, Train Clean=green, Train Noisy=red
    - ACC/F1 shown in percent (0–100)
    - No left/right padding on x-axis
    """
    if len(json_paths) != 4:
        raise ValueError("Provide exactly 4 JSON paths.")
    if len(labels) != 4:
        raise ValueError("Provide exactly 4 labels (one per JSON).")

    parsed = []
    for p in json_paths:
        p = Path(p)
        with p.open("r") as f:
            d = json.load(f)

        alpha_items = []
        for k, v in d.items():
            alpha_items.append((float(k), v))
        alpha_items.sort(key=lambda x: x[0])

        alphas = [a for a, _ in alpha_items]
        test_vals, clean_vals, noisy_vals = [], [], []
        for _, entry in alpha_items:
            test_vals.append(entry["test_results"][metric_key])
            clean_vals.append(entry["train_results"]["clean_set"][metric_key])
            noisy_vals.append(entry["train_results"]["noisy_set"][metric_key])
        parsed.append((alphas, test_vals, clean_vals, noisy_vals))

    # Global alpha bounds (so all subplots line up) + remove x padding
    global_alpha_min = min(min(a) for a, *_ in parsed)
    global_alpha_max = max(max(a) for a, *_ in parsed)

    scale_to_percent = metric_key.upper() in {"ACC", "F1"}
    scale = 100.0 if scale_to_percent else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    legend_handles = None
    for ax, (title, series) in zip(axes, zip(labels, parsed)):
        alphas, test_vals, clean_vals, noisy_vals = series

        h1, = ax.plot(alphas, [v * scale for v in test_vals], label='Test Accuracy', color='tab:blue')
        h2, = ax.plot(alphas, [v * scale for v in clean_vals], label='Train Clean Accuracy', color='tab:green')
        h3, = ax.plot(alphas, [v * scale for v in noisy_vals], label='Train Noisy Accuracy', color='tab:red')

        ax.set_title(title)
        ax.set_xlabel(r'$\alpha$')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

        # ⬇️ key lines to remove the left/right gaps
        ax.set_xlim(global_alpha_min, global_alpha_max)  # pin to endpoints
        ax.margins(x=0)  # or: ax.set_xmargin(0)

        if scale_to_percent:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])

        if legend_handles is None:
            legend_handles = [h1, h2, h3]

    fig.supylabel('Accuracy (%)' if scale_to_percent else metric_key)

    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc='lower center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0)
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=200)
    if show:
        plt.show()

    return fig, axes



def plot_acc_curves_from_jsons_1x2(json_paths, labels, metric_key="ACC", savepath=None, show=True):
    """
    Plots accuracy curves from 2 JSON files in a 1x2 grid.

    - No markers (simple lines)
    - Colors: Test=blue, Train Clean=green, Train Noisy=red
    - ACC/F1 shown in percent (0–100)
    - No left/right padding on x-axis
    """
    # --- MODIFIED: Expect exactly 2 JSON paths and labels ---
    if len(json_paths) != 2:
        raise ValueError("Provide exactly 2 JSON paths.")
    if len(labels) != 2:
        raise ValueError("Provide exactly 2 labels (one per JSON).")

    parsed = []
    for p in json_paths:
        p = Path(p)
        with p.open("r") as f:
            d = json.load(f)

        alpha_items = []
        for k, v in d.items():
            alpha_items.append((float(k), v))
        alpha_items.sort(key=lambda x: x[0])

        alphas = [a for a, _ in alpha_items]
        test_vals, clean_vals, noisy_vals = [], [], []
        for _, entry in alpha_items:
            test_vals.append(entry["test_results"][metric_key])
            clean_vals.append(entry["train_results"]["clean_set"][metric_key])
            noisy_vals.append(entry["train_results"]["noisy_set"][metric_key])
        parsed.append((alphas, test_vals, clean_vals, noisy_vals))

    # Global alpha bounds (so all subplots line up) + remove x padding
    global_alpha_min = min(min(a) for a, *_ in parsed)
    global_alpha_max = max(max(a) for a, *_ in parsed)

    scale_to_percent = metric_key.upper() in {"ACC", "F1"}
    scale = 100.0 if scale_to_percent else 1.0

    # --- MODIFIED: Create a 1x2 subplot grid ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    # axes is already a 1D array of length 2, so .ravel() is not needed but harmless

    legend_handles = None
    for ax, (title, series) in zip(axes, zip(labels, parsed)):
        alphas, test_vals, clean_vals, noisy_vals = series

        h1, = ax.plot(alphas, [v * scale for v in test_vals], label='Test Accuracy', color='tab:blue')
        h2, = ax.plot(alphas, [v * scale for v in clean_vals], label='Train Clean Accuracy', color='tab:green')
        h3, = ax.plot(alphas, [v * scale for v in noisy_vals], label='Train Noisy Accuracy', color='tab:red')

        ax.set_title(title)
        ax.set_xlabel(r'$\alpha$')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

        # ⬇️ key lines to remove the left/right gaps
        ax.set_xlim(global_alpha_min, global_alpha_max)
        ax.margins(x=0)

        if scale_to_percent:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])

        if legend_handles is None:
            legend_handles = [h1, h2, h3]

    fig.supylabel('Accuracy (%)' if scale_to_percent else metric_key)

    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc='lower center',
        ncol=3,
        frameon=False,
        # bbox_to_anchor=(0.5, 0.05) # Adjusted anchor for better spacing
    )
    # Adjusted rect to give more space for the bottom legend
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
    if show:
        plt.show()

    return fig, axes


# jsons = [
#     "results/single_experiment/clip_noise_TA/config26/metrics_mtl.json",
#     "results/single_experiment/clip_noise_TA/config28/metrics_mtl.json",
#     "results/single_experiment/clip_noise_TA/config40/metrics_mtl.json",
#     "results/single_experiment/clip_noise_TA/config41/metrics_mtl.json"
# ]
# labels = [r"CIFAR-10 ($\eta=40\%$)", r"CIFAR-10 ($\eta=60\%$)", r"MNIST ($\eta=60\%$)", r"MNIST ($\eta=80\%$)"]
# plot_acc_curves_from_jsons(jsons, labels, savepath='visulaization_dir/clip_mix_interpolation_curves.png') 


jsons = [
    "results/single_experiment/clip_noise_TA/config28/metrics_mtl.json",
    "results/single_experiment/clip_noise_TA/config41/metrics_mtl.json"
]
labels = [r"CIFAR-10 ($\eta=60\%$)", r"MNIST ($\eta=80\%$)"]
plot_acc_curves_from_jsons_1x2(jsons, labels, savepath='visulaization_dir/clip_mix_interpolation_curves_paper.png') 