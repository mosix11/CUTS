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
        fig.savefig(savepath, dpi=500)

    if show:
        plt.show()
    plt.close(fig)



def plot_tv_similarity_grid_2x3(
    pickle_paths,
    subtitles,
    tv_names,
    cmap="vlag",
    vmin=-1.0,
    vmax=1.0,
    figsize=(12.5, 7.6),
    tick_label_font_size=9,
    annot=True,
    savepath=None,
    show=True,
):
    """
    Plot six 6x6 cosine-similarity matrices in a 2x3 grid with a single colorbar.

    Args:
        pickle_paths (list[str|Path]): Paths to SIX pickles storing 6x6 matrices
            (numpy arrays, torch tensors, or dicts with keys like 'sim', 'matrix', ...).
        subtitles (list[str]): Six captions shown under each matrix.
        tv_names (list[str]): Tick labels for both axes (length should be 6).
        cmap (str): Colormap. Default 'vlag'.
        vmin, vmax (float): Shared color range. Default [-1, 1].
        figsize (tuple): Figure size.
        tick_label_font_size (int): Tick label font size.
        annot (bool): If True, show numeric annotations.
        savepath (str|Path|None): If provided, save to this path.
        show (bool): If True, display the figure.
    """
    # --- checks ---
    if len(pickle_paths) != 6:
        raise ValueError("Provide exactly six pickle paths.")
    if len(subtitles) != 6:
        raise ValueError("Provide exactly six subtitles.")
    if len(tv_names) != 6:
        raise ValueError("tv_names must have length 6 for 6x6 matrices.")

    # --- load matrices ---
    mats = []
    for p in pickle_paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        # Accept numpy arrays, torch tensors, or dicts containing a matrix
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                arr = obj.detach().cpu().numpy()
            else:
                arr = None
        except Exception:
            arr = None
        if arr is None:
            if isinstance(obj, np.ndarray):
                arr = obj
            elif isinstance(obj, dict):
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
                if arr is None:
                    raise ValueError(f"Dict in {p} did not contain an obvious matrix key.")
            else:
                arr = np.asarray(obj)

        if arr.shape != (6, 6):
            raise ValueError(f"Matrix in {p} has shape {arr.shape}, expected (6, 6).")

        mats.append(arr.astype(float))

    # --- plot ---
    sns.set(style="white")
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for ax, mat, subtitle in zip(axes, mats, subtitles):
        kwargs = dict(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,      # single shared cbar added later
            xticklabels=tv_names,
            yticklabels=tv_names,
            square=True,
        )
        if annot:
            kwargs.update(annot=True, fmt=".2f")
        sns.heatmap(mat, ax=ax, **kwargs)

        # Style ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=tick_label_font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center",
                           fontsize=tick_label_font_size)

        # No axis labels; use x-label area for the caption
        ax.set_xlabel(subtitle, labelpad=10, fontsize=tick_label_font_size + 1)
        ax.set_ylabel(None)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=tick_label_font_size)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300)

    if show:
        plt.show()
    plt.close(fig)


# pickle_paths_pois = [
#     '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config1/confusion_mats/tv_sim.pkl',
#     # '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config2/confusion_mats/tv_sim.pkl',
#     '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config3/confusion_mats/tv_sim.pkl',
#     '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_poison_TA/config1/confusion_mats/tv_sim.pkl',
#     '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config2/confusion_mats/tv_sim.pkl'
# ]

pickle_paths_asym = [
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_noise_TA/config42/confusion_mats/tv_sim.pkl',
    # '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_poison_TA/config2/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_noise_TA/config43/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_noise_TA/config4/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_noise_TA/config5/confusion_mats/tv_sim.pkl'
]

tv_names = [r"$\tau_p$", r"$\tau_\text{CF}$", r"$\tau_r$", r"$\tau_\text{mix}$"]
# subtitles_pois = ['CLIP—MNIST', 'CLIP—CIFAR-100', 'DINO—CIFAR-10', 'DINO—CIFAR-100']
subtitles_asym = ['CLIP—CIFAR-10', 'CLIP—CIFAR-100', 'DINO—CIFAR-10', 'DINO—CIFAR-100']

# plot_tv_similarity_grid(
#     pickle_paths_asym,
#     subtitles_asym,
#     tv_names,
#     cmap="vlag",
#     vmin=-1.0,
#     vmax=1.0,
#     figsize=(12.5, 3.6),
#     tick_label_font_size=10,
#     savepath="visulaization_dir/task_sim_grid_asym.png",
#     show=True,
# )



pickle_paths_sym = [
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_noise_TA/config26/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_noise_TA/config27/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/clip_noise_TA/config39/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_noise_TA/config1/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_noise_TA/config2/confusion_mats/tv_sim.pkl',
    '/home/mosix11/Projects/TaskVectors/files/results/single_experiment/dino_noise_TA/config3/confusion_mats/tv_sim.pkl'
]

tv_names = [r"$\tau_p^1$", r"$\tau_p^2$", r"$\tau_p^\text{avg}$", r"$\tau_\text{CF}$", r"$\tau_r$", r"$\tau_\text{mix}$"]
subtitles_asym = ['CLIP—CIFAR-10', 'CLIP—CIFAR-100', 'DINO—CIFAR-10', 'DINO—CIFAR-100']