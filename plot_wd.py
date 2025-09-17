import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_wd_pickle(
    pickle_path: str,
    *,
    tv1_label: str = r"$\alpha_c$",     # x-axis label (taskvector1)
    tv2_label: str = r"$\alpha_t$",     # y-axis label (taskvector2)
    title: str | None = None,
    percent: bool = True,               # show 0–100% like the paper
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Load a pickled dict returned by apply_WD_analysis and plot the heatmap.

    Expects keys: 'alphas', 'wd_map', optionally 'best' with
    {'alpha_tv1': float, 'alpha_tv2': float, 'wd': float}.
    """
    with open(pickle_path, "rb") as f:
        out = pickle.load(f)

    alphas = np.asarray(out["alphas"], dtype=float)          # 1D
    wd_map = np.asarray(out["wd_map"], dtype=float)          # [len(a2), len(a1)]

    # Convert to % if requested
    if percent:
        Z = 100.0 * wd_map
        vmin, vmax = 0.0, 100.0
        cbar_label = r"$\xi(\alpha_c,\alpha_t)$ [%]"
    else:
        Z = wd_map
        vmin, vmax = 0.0, 1.0
        cbar_label = r"$\xi(\alpha_c,\alpha_t)$"

    # Build cell edges so ticks align with alpha values
    if len(alphas) > 1:
        step = float(np.mean(np.diff(alphas)))
    else:
        step = 1.0
    edges = np.r_[alphas[0] - step/2, alphas + step/2]       # length N+1

    Xe, Ye = np.meshgrid(edges, edges)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=130)

    # pcolormesh with 'edges' puts the first row/col at the bottom-left (good)
    hm = ax.pcolormesh(Xe, Ye, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(tv1_label)          # x: α for taskvector1 (“clean” in the paper)
    ax.set_ylabel(tv2_label)          # y: α for taskvector2 (“triggered”)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])

    # Manage ticks so they don’t get crowded
    if len(alphas) > 1:
        tick_every = max(1, len(alphas)//8)
        ticks = alphas[::tick_every]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # Title
    if title is None:
        title = r"$\xi(\alpha_c,\alpha_t)$"
    ax.set_title(title)

    # Mark the minimum (best) point
    best = out.get("best")
    if best is None:
        iy, ix = np.unravel_index(np.argmin(wd_map), wd_map.shape)
        best_alpha_tv1 = alphas[ix]
        best_alpha_tv2 = alphas[iy]
    else:
        best_alpha_tv1 = float(best["alpha_tv1"])
        best_alpha_tv2 = float(best["alpha_tv2"])
    ax.plot([best_alpha_tv1], [best_alpha_tv2],
            marker="o", markersize=4, markeredgecolor="white",
            markerfacecolor="none", linewidth=0)

    # Colorbar
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax



plot_wd_pickle(
    "results/single_experiment/clip_noise_TA/config28/WD.pkl",
    tv1_label=r"$\alpha_c$",
    tv2_label=r"$\alpha_t$",
    title=r"$\xi(\alpha_c,\alpha_t)$ — CIFAR-10",
    percent=True,
    cmap="viridis",
    save_path="wd_heatmap.png",
)