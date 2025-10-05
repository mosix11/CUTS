import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_wd_pickle(
    pickle_path: str,
    *,
    tv1_label: str = r"$\alpha_{\text{clean}}$",     # x-axis label (taskvector1)
    tv2_label: str = r"$\alpha_{\text{corruption}}$",     # y-axis label (taskvector2)
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



def plot_antitask_wd_maps(pickle_path):
    """
    Plot key 2D maps from apply_WD_antitask_analysis() in one figure.

    Expected keys in `res`:
      'alphas', 'risk_base', 'risk_c_only', 'risk_n_only',
      'risk_map', 'delta_c', 'delta_n', 'interaction', 'wd_map', 'best'
    """
    
    with open(pickle_path, "rb") as f:
        res = pickle.load(f)
            
    alphas = np.asarray(res["alphas"], dtype=float)
    risk_base = float(res["risk_base"])
    risk_c_only = np.asarray(res["risk_c_only"], dtype=float)   # [W]
    risk_n_only = np.asarray(res["risk_n_only"], dtype=float)   # [H]
    risk_map = np.asarray(res["risk_map"], dtype=float)         # [H,W]
    delta_c = np.asarray(res["delta_c"], dtype=float)           # [W]
    delta_n = np.asarray(res["delta_n"], dtype=float)           # [H]
    interaction = np.asarray(res["interaction"], dtype=float)   # [H,W]
    wd_map = np.asarray(res["wd_map"], dtype=float)             # [H,W]
    best = res.get("best", None)
    
    # print(risk_n_only - risk_base)
    # print(risk_map[:, 20] - risk_base + risk_c_only)
    # # exit()

    H, W = risk_map.shape
    assert H == len(alphas) and W == len(alphas), "Maps should be len(alphas)×len(alphas)."

    # Build an additive (no-interaction) risk prediction and the normalization denom map
    # \hat R_add(αc, αn) = R(αc,0) + R(0,αn) - R(0,0)
    R_add = (risk_n_only[:, None] + risk_c_only[None, :] - risk_base).astype(np.float32)
    denom = (np.abs(delta_n)[:, None] + np.abs(delta_c)[None, :]).astype(np.float32)  # |Δc|+|Δn|
    
    
    # print(risk_base)
    # print((risk_map - risk_c_only)[:, 20])
    # exit()

    # Nice aligned edges for pcolormesh so ticks sit on alpha values
    if len(alphas) > 1:
        step = float(np.mean(np.diff(alphas)))
    else:
        step = 1.0
    edges = np.r_[alphas[0] - step/2, alphas + step/2]
    Xe, Ye = np.meshgrid(edges, edges)

    # Helpers
    def _heat(ax, Z, title, cbar_label=None, vmin=None, vmax=None, mark_best=False):
        hm = ax.pcolormesh(Xe, Ye, Z, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$\alpha_c$")  # x-axis: clean vector scale
        ax.set_ylabel(r"$\alpha_n$")  # y-axis: noise vector scale
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[0], edges[-1])
        ax.set_title(title)
        # tick thinning to avoid clutter
        if len(alphas) > 1:
            tick_every = max(1, len(alphas)//8)
            ticks = alphas[::tick_every]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        cbar = plt.colorbar(hm, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)
        if mark_best and best is not None:
            ax.plot([best["alpha_c"]], [best["alpha_n"]],
                    marker="o", markersize=4, markeredgecolor="white",
                    markerfacecolor="none", linewidth=0)

    # Figure layout: 2 rows × 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), dpi=130)
    ax11, ax12, ax13 = axs[0]
    ax21, ax22, ax23 = axs[1]

    # Choose color ranges for comparability where it helps
    # (risk-like maps share a common vmin/vmax)
    risk_like = np.concatenate([risk_map.ravel(), R_add.ravel()])
    vmin_risk = float(np.nanmin(risk_like))
    vmax_risk = float(np.nanmax(risk_like))

    _heat(ax11, risk_map,          r"Measured risk $R(\alpha_c,\alpha_n)$",
          cbar_label="Risk", vmin=vmin_risk, vmax=vmax_risk, mark_best=False)

    _heat(ax12, R_add,             r"Additive prediction $\hat R_{\mathrm{add}}$",
          cbar_label="Risk", vmin=vmin_risk, vmax=vmax_risk, mark_best=False)

    # Interaction can be positive/negative; center colormap around 0
    max_abs_I = float(np.nanmax(np.abs(interaction)))
    _heat(ax13, interaction,       r"Interaction $I=R-\hat R_{\mathrm{add}}$",
          cbar_label="Interaction", vmin=-max_abs_I, vmax=max_abs_I, mark_best=False)

    # Denominator map used in ξ normalization (|Δc|+|Δn|)
    _heat(ax21, denom,             r"Normalization map $|\Delta_c|+|\Delta_n|$",
          cbar_label="Magnitude", vmin=0.0, vmax=float(np.nanmax(denom)), mark_best=False)

    # Normalized disentanglement error ξ_anti (lower is better)
    _heat(ax22, wd_map,            r"Normalized WD $\xi_{\mathrm{anti}}$",
          cbar_label=r"$\xi_{\mathrm{anti}}$", vmin=0.0, vmax=float(np.nanmax(wd_map)), mark_best=True)

    # Residual = measured - additive (another way to show interaction sign/scale)
    # (Duplicate of interaction numerically, but sometimes nice to see with different scaling.)
    _heat(ax23, np.abs(risk_map - R_add),  r"Abs Interaction $|I|$",
          cbar_label="Interaction (abs)", vmin=0, vmax=max_abs_I, mark_best=False)

    fig.suptitle("Task vs. Anti-task Weight Disentanglement Maps", y=0.995, fontsize=12)
    fig.tight_layout()
    return fig, axs


# plot_wd_pickle(
#     # "results/single_experiment/clip_noise_TA/config28/WD.pkl",
#     # "results/single_experiment/clip_poison_TA/config2/WD.pkl",
#     "results/single_experiment/clip_noise_TA/config7/WD.pkl",
#     tv1_label=r"$\alpha_c$",
#     tv2_label=r"$\alpha_t$",
#     title=r"$\xi(\alpha_c,\alpha_t)$ — CIFAR-10",
#     percent=True,
#     cmap="viridis",
#     save_path="wd_heatmap3.png",
# )


# fig, axs = plot_wd_pickle('results/single_experiment/clip_poison_TA/config1/WD2.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/regular_poison_TA/config4/WD2.pkl')

# fig, axs = plot_wd_pickle('results/single_experiment/clip_poison_TA/config2/WD2.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/regular_poison_TA/config2/WD2.pkl')

fig, axs = plot_wd_pickle('results/single_experiment/clip_poison_TA/config3/WD2.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/regular_poison_TA/config3/WD2.pkl')

# fig, axs = plot_wd_pickle('results/single_experiment/clip_noise_TA/config26/WD_AT2_acc_real.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/regular_noise_TA/config23/WD2.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/regular_noise_TA/config34/WD2.pkl')

# fig, axs = plot_wd_pickle('results/single_experiment/regular_noise_TA/config18/WD2.pkl')
# fig, axs = plot_wd_pickle('results/single_experiment/clip_noise_TA/config39/WD2.pkl')

# fig, axs = plot_antitask_wd_maps('results/single_experiment/clip_noise_TA/config26/WD_AT2_acc.pkl')
# fig, axs = plot_antitask_wd_maps('results/single_experiment/clip_noise_TA/config7/WD_AT2.pkl') 
plt.show()