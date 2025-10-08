import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tv_similarity_grid(
    pickle_paths,
    subtitles,
    tv_names,
    cmap="vlag",
    vmin=-1.0,
    vmax=1.0,
    figsize=(12, 3.4),
    tick_label_font_size=10,
    savepath=None,
    show=True,
):
    """
    Plot four 4x4 cosine-similarity matrices in one row with a single colorbar.

    Args:
        pickle_paths (list[str|Path]): Paths to FOUR pickles storing 4x4 matrices (np.ndarray or torch.Tensor).
        subtitles (list[str]): Four short labels, one per matrix, shown *under* each heatmap.
        tv_names (list[str]): Tick labels (e.g., [r"$\\tau_p$", r"$\\tau_\\text{CF}$", r"$\\tau_\\text{mix}$", r"$\\tau_r$"]).
        cmap (str): Matplotlib/Seaborn colormap. Default "vlag".
        vmin, vmax (float): Shared color range, defaults to [-1, 1].
        figsize (tuple): Figure size.
        tick_label_font_size (int): Font size for tick labels.
        savepath (str|Path|None): If given, saves the figure there.
        show (bool): If True, displays the figure.
    """
    # --- load and normalize matrices ---
    mats = []
    for p in pickle_paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        # Accept numpy arrays, torch tensors, or dicts that contain an obvious matrix
        try:
            import torch
            is_torch = isinstance(obj, torch.Tensor)
        except Exception:
            is_torch = False

        if is_torch:
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            arr = obj
        elif isinstance(obj, dict):
            # Try a few common keys
            for k in ("cm", "sim", "similarity", "matrix", "task_sim"):
                if k in obj:
                    val = obj[k]
                    try:
                        import torch
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                    except Exception:
                        pass
                    arr = np.asarray(val)
                    break
            else:
                raise ValueError(f"Dict in {p} did not contain an obvious matrix key.")
        else:
            arr = np.asarray(obj)

        if arr.shape != (4, 4):
            raise ValueError(f"Matrix in {p} has shape {arr.shape}, expected (4, 4).")
        mats.append(arr.astype(float))

    if len(mats) != 4:
        raise ValueError("Provide exactly four pickle paths.")

    if len(subtitles) != 4:
        raise ValueError("Provide exactly four subtitles (one per matrix).")

    # --- plotting ---
    sns.set(style="white")
    fig, axes = plt.subplots(
        1, 4, figsize=figsize, constrained_layout=True, squeeze=False
    )
    axes = axes[0]

    # Shared normalization for all panels
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, (ax, mat, subtitle) in enumerate(zip(axes, mats, subtitles)):
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,      # we'll add a single shared colorbar later
            annot=True,
            fmt=".2f",
            xticklabels=tv_names,
            yticklabels=tv_names,
            square=True,
        )
        # Tick label styling
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_label_font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=tick_label_font_size)

        # No axis labels; use x-label area for the experiment caption
        ax.set_xlabel(subtitle, labelpad=10, fontsize=tick_label_font_size + 1)
        ax.set_ylabel(None)

    # One shared colorbar on the right
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older Matplotlibs
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=tick_label_font_size)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300)

    if show:
        plt.show()
    plt.close(fig)




pickle_paths = [
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config1/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config2/confusion_mats/tv_sim.pkl',
    # '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config3/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_poison_TA/config1/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config2/confusion_mats/tv_sim.pkl'
]

tv_names = [r"$\tau_p$", r"$\tau_\text{CF}$", r"$\tau_\text{mix}$", r"$\tau_r$"]