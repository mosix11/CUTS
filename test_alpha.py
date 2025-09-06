import torch
import math
from typing import Dict, List, Tuple
from collections import defaultdict

@torch.no_grad()
def _reshape_weight_matrix(W: torch.Tensor, is_conv: bool) -> torch.Tensor:
    """
    Returns a 2D matrix for spectral/Gram computations.
    For conv: (out_channels, in_channels * kH * kW)
    For linear: already 2D (out, in)
    """
    if is_conv:
        oc, ic, kh, kw = W.shape
        return W.view(oc, ic * kh * kw)
    else:
        return W  # shape (out, in)

@torch.no_grad()
def _power_iteration_spectral_norm(W2d: torch.Tensor, iters: int = 20, eps: float = 1e-12) -> float:
    """
    Spectral norm (largest singular value) via power iteration on W^T W.
    Works on CPU/GPU tensors.
    """
    device = W2d.device
    # operate on smaller side for speed
    if W2d.shape[0] >= W2d.shape[1]:
        n = W2d.shape[1]
        v = torch.randn(n, device=device)
        v = v / (v.norm() + eps)
        for _ in range(iters):
            u = W2d @ v
            u = u / (u.norm() + eps)
            v = W2d.t() @ u
            v = v / (v.norm() + eps)
        sigma = (u @ (W2d @ v)).item()
    else:
        n = W2d.shape[0]
        u = torch.randn(n, device=device)
        u = u / (u.norm() + eps)
        for _ in range(iters):
            v = W2d.t() @ u
            v = v / (v.norm() + eps)
            u = W2d @ v
            u = u / (u.norm() + eps)
        sigma = (u @ (W2d @ v)).item()
    return max(sigma, 0.0)

@torch.no_grad()
def _coherence_metrics(W2d: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Compute two orthogonality metrics using *row vectors* as filters (out-dims).
    1) mutual coherence (mean absolute off-diagonal of normalized Gram)
    2) off-diagonal Frobenius energy of normalized Gram
    """
    # Normalize rows to unit norm
    norms = W2d.norm(dim=1, keepdim=True).clamp_min(eps)
    U = W2d / norms
    G = U @ U.t()  # (out, out)
    # zero diagonal
    G_off = G - torch.diag_embed(torch.diag(G))
    mut_coh = G_off.abs().mean().item()
    offdiag_energy = (G_off ** 2).sum().sqrt().item()
    return mut_coh, offdiag_energy

@torch.no_grad()
def _bn_alpha_safe_max(state0: Dict[str, torch.Tensor],
                       dstate: Dict[str, torch.Tensor],
                       eps: float = 1e-5) -> float:
    """
    Ensure running_var(α) >= eps for all BN buffers that appear in the delta.
    running_var(α) = var0 - α * Δvar
    Returns +inf if no BN buffers in delta.
    """
    amax = float('inf')
    for k, t0 in state0.items():
        if 'running_var' in k and k in dstate:
            dv = dstate[k]
            # var(α) = t0 - α * dv  >= eps
            # If dv > 0, α <= (t0 - eps)/dv; if dv <= 0, no constraint
            with torch.no_grad():
                dv_pos_mask = dv > 0
                if dv_pos_mask.any():
                    bound = ((t0[dv_pos_mask] - eps) / dv[dv_pos_mask]).min().item()
                    amax = min(amax, float(max(bound, 0.0)))
    return amax

@torch.no_grad()
def pick_alpha_weight_only(
    state0: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    alphas: List[float] = None,
    lambda_lip: float = 0.5,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """
    Weight-only alpha selector.
    Inputs:
      - state0: base model state_dict (tensors only)
      - delta:  task vector (same keys, tensors)
      - alphas: grid to scan; default builds [0, α_bn_safe] with 41 points
    Returns dict with:
      - alpha_best, curves, per-layer diagnostics
    """
    # Move to device (in-place copies)
    s0 = {k: v.to(device) for k, v in state0.items()}
    d  = {k: v.to(device) for k, v in delta.items() if k in state0}

    # BN guardrail
    amax_bn = _bn_alpha_safe_max(s0, d, eps=1e-5)
    if alphas is None:
        if math.isfinite(amax_bn) and amax_bn > 0:
            hi = max(1e-6, min(2.0, 0.95 * amax_bn))
        else:
            hi = 2.0
        alphas = torch.linspace(0.0, hi, steps=41).tolist()

    # Collect layer keys for Linear/Conv weights we will score
    layer_keys = []
    for k, t in s0.items():
        if not k.endswith('.weight'):
            continue
        # Identify Linear/Conv by tensor rank: Linear->2D, Conv{1,2,3}d->4/5D
        if t.ndim == 2 or t.ndim == 4 or t.ndim == 5:
            layer_keys.append(k)

    # Curves across alphas
    curves = defaultdict(list)  # 'mut_coh', 'offdiag_energy', 'lip_sumlog'
    per_layer = {k: {'mut_coh': [], 'offdiag_energy': [], 'specnorm': []} for k in layer_keys}

    for a in alphas:
        lip_sumlog = 0.0
        mut_coh_all = []
        offdiag_all = []

        for k in layer_keys:
            W0 = s0[k]
            dW = d.get(k, torch.zeros_like(W0))
            W  = W0 - a * dW

            is_conv = (W.ndim >= 4)
            W2d = _reshape_weight_matrix(W, is_conv)

            # Spectral norm (operator norm)
            sigma = _power_iteration_spectral_norm(W2d, iters=20)
            lip_sumlog += math.log(max(sigma, 1e-12))

            # Coherence metrics on filters (rows)
            mut_coh, offdiag_energy = _coherence_metrics(W2d)
            mut_coh_all.append(mut_coh)
            offdiag_all.append(offdiag_energy)

            per_layer[k]['specnorm'].append(sigma)
            per_layer[k]['mut_coh'].append(mut_coh)
            per_layer[k]['offdiag_energy'].append(offdiag_energy)

        # Aggregate across layers
        curves['lip_sumlog'].append(lip_sumlog)
        curves['mut_coh_mean'].append(float(sum(mut_coh_all) / len(mut_coh_all)))
        curves['offdiag_energy_mean'].append(float(sum(offdiag_all) / len(offdiag_all)))

    # Z-score each curve over alpha
    def zscore(xs):
        xs = torch.tensor(xs)
        m, s = xs.mean(), xs.std(unbiased=False).clamp_min(1e-12)
        return ((xs - m) / s).tolist()

    z_mut = zscore(curves['mut_coh_mean'])
    z_off = zscore(curves['offdiag_energy_mean'])
    z_lip = zscore(curves['lip_sumlog'])

    # Composite score (lower is better). We’ll minimize it.
    composite = [ (z_off[i] + z_mut[i] + lambda_lip * z_lip[i]) for i in range(len(alphas)) ]
    best_idx = int(torch.tensor(composite).argmin().item())
    alpha_best = alphas[best_idx]

    return {
        'alpha_best': alpha_best,
        'alphas': alphas,
        'curves': {
            'mut_coh_mean': curves['mut_coh_mean'],
            'offdiag_energy_mean': curves['offdiag_energy_mean'],
            'lip_sumlog': curves['lip_sumlog'],
            'composite': composite,
            'alpha_bn_max': amax_bn,
        },
        'per_layer': per_layer,  # per-layer curves if you want to inspect elbows layerwise
    }
    
    
    
import math
import matplotlib.pyplot as plt

def plot_weight_only_curves(res, title_prefix="Alpha sweep (weight-only)"):
    """
    res: the dict returned by pick_alpha_weight_only(...)
         expected keys: 'alpha_best', 'alphas', 'curves' (with
         'mut_coh_mean', 'offdiag_energy_mean', 'lip_sumlog', 'composite', 'alpha_bn_max')
    """
    alphas = res['alphas']
    curves = res['curves']
    alpha_best = res['alpha_best']
    alpha_bn_max = curves.get('alpha_bn_max', float('inf'))

    def _vline_alpha(ax):
        ax.axvline(alpha_best, linestyle='--', linewidth=1.5)
        if math.isfinite(alpha_bn_max):
            ax.axvline(alpha_bn_max, linestyle=':', linewidth=1.0)
        ax.set_xlabel(r'$\alpha$')
        ax.grid(True, linestyle=':', linewidth=0.5)

    # 1) Composite
    plt.figure()
    plt.plot(alphas, curves['composite'])
    _vline_alpha(plt.gca())
    plt.title(f"{title_prefix} – Composite score (lower is better)")
    plt.ylabel("z(offdiag) + z(mutcoh) + λ·z(lip)")

    # 2) Mutual coherence (mean abs off-diag of normalized Gram)
    plt.figure()
    plt.plot(alphas, curves['mut_coh_mean'])
    _vline_alpha(plt.gca())
    plt.title(f"{title_prefix} – Mutual coherence (lower is better)")
    plt.ylabel("Mean |off-diag| (normalized Gram)")

    # 3) Off-diagonal energy (normalized Gram)
    plt.figure()
    plt.plot(alphas, curves['offdiag_energy_mean'])
    _vline_alpha(plt.gca())
    plt.title(f"{title_prefix} – Off-diag energy (lower is better)")
    plt.ylabel("Frobenius norm of off-diag")

    # 4) Lipschitz proxy (sum of log spectral norms across layers)
    plt.figure()
    plt.plot(alphas, curves['lip_sumlog'])
    _vline_alpha(plt.gca())
    plt.title(f"{title_prefix} – Lipschitz proxy (lower is better)")
    plt.ylabel("Σ log σ_max (per layer)")

    plt.show()

# def plot_per_layer_curves(res, layer_key_substr=None, max_layers=12, title_prefix="Per-layer"):
#     """
#     Optional: visualize per-layer spec norm / coherence curves.
#     - layer_key_substr: only plot layers whose key contains this substring (e.g., 'layer4' or 'fc')
#     - max_layers: safety cap to avoid flooding plots if model is huge
#     """
#     alphas = res['alphas']
#     per_layer = res['per_layer']
#     alpha_best = res['alpha_best']
#     alpha_bn_max = res['curves'].get('alpha_bn_max', float('inf'))

#     def _vline(ax):
#         ax.axvline(alpha_best, linestyle='--', linewidth=1.0)
#         if math.isfinite(alpha_bn_max):
#             ax.axvline(alpha_bn_max, linestyle=':', linewidth=1.0)
#         ax.set_xlabel(r'$\alpha$')
#         ax.grid(True, linestyle=':', linewidth=0.5)

#     # Filter layers
#     keys = [k for k in per_layer.keys() if (layer_key_substr in k) if layer_key_substr else True]
#     keys = keys[:max_layers]

#     # Spec norm per layer
#     for k in keys:
#         plt.figure()
#         plt.plot(alphas, per_layer[k]['specnorm'])
#         _vline(plt.gca())
#         plt.title(f"{title_prefix} – σ_max: {k}")
#         plt.ylabel("Spectral norm")

#     # Mutual coherence per layer
#     for k in keys:
#         plt.figure()
#         plt.plot(alphas, per_layer[k]['mut_coh'])
#         _vline(plt.gca())
#         plt.title(f"{title_prefix} – Mutual coherence: {k}")
#         plt.ylabel("Mean |off-diag| (normalized Gram)")

#     # Off-diag energy per layer
#     for k in keys:
#         plt.figure()
#         plt.plot(alphas, per_layer[k]['offdiag_energy'])
#         _vline(plt.gca())
#         plt.title(f"{title_prefix} – Off-diag energy: {k}")
#         plt.ylabel("Frobenius norm of off-diag")

#     plt.show()