# Compute Recovery Rates for the "Full results for FC1 / ResNet18" table

mix = [
    # MNIST+FC1: SN (10,20,40,60,80), AN (40)
    96.5, 92.9, 79.7, 56.2, 26.3, 81.9,
    # CIFAR10+ResNet18: SN (10,20,40,60,80), AN (40)
    83.4, 77.9, 60.0, 38.8, 16.6, 74.5,
    # CIFAR100+ResNet18: SN (10,20,40,60,80), AN (40)
    61.0, 52.6, 38.3, 20.5, 6.7, 41.5,
]

clean = [
    # MNIST+FC1
    98.5, 98.4, 98.3, 97.9, 97.1, 98.3,
    # CIFAR10+ResNet18
    89.7, 88.9, 87.4, 85.0, 79.5, 88.7,
    # CIFAR100+ResNet18
    68.1, 66.9, 63.9, 57.5, 45.4, 63.9,
]

mix_tau = [
    # MNIST+FC1
    95.6, 91.9, 79.0, 55.5, 26.2, 81.9,
    # CIFAR10+ResNet18
    75.9, 66.0, 9.9, 10.1, 10.2, 24.3,
    # CIFAR100+ResNet18
    44.2, 1.2, 1.0, 0.9, 2.1, 1.5,
]

cf = [
    # MNIST+FC1
    96.4, 94.9, 91.1, 87.5, 80.1, 90.9,
    # CIFAR10+ResNet18
    83.6, 80.0, 69.2, 57.2, 32.0, 80.0,
    # CIFAR100+ResNet18
    60.9, 53.5, 41.8, 26.1, 11.4, 45.8,
]

sap = [
    # MNIST+FC1
    97.1, 96.4, 93.8, 91.0, 69.1, 86.9,
    # CIFAR10+ResNet18
    83.8, 81.2, 67.1, 48.4, 19.9, 75.0,
    # CIFAR100+ResNet18
    61.4, 53.1, 40.7, 24.0, 8.7, 42.0,
]

uhat = [
    # MNIST+FC1
    96.5, 95.0, 93.6, 88.4, 71.4, 93.6,
    # CIFAR10+ResNet18
    84.3, 81.2, 65.8, 52.7, 19.5, 73.8,
    # CIFAR100+ResNet18
    60.2, 54.4, 42.1, 23.3, 7.9, 44.5,
]

def rr(model, mix, clean):
    vals = []
    for m, x, c in zip(model, mix, clean):
        denom = (c - x)
        num = (m - x)
        if denom == 0:
            # Avoid division by zero; define RR as 0 if numerator is also 0, else +/- inf.
            rr_val = float('nan') if num != 0 else 0.0
        else:
            rr_val = (num / denom) * 100.0
        vals.append(rr_val)
    return vals

def latex_row(name, vals, group_sizes=(6,6,6)):
    # Insert an empty column between dataset groups
    out = []
    idx = 0
    for g in group_sizes:
        group_vals = vals[idx:idx+g]
        for v in group_vals:
            # if v != v:  # NaN check
            #     out.append(r"\textit{N/A}")
            # elif v < 0:
            #     out.append(r"\textit{Fail}")
            # else:
            out.append(f"{round(v, 1):.1f}")
        idx += g
        if idx < len(vals):
            out.append("")  # blank cell for vertical bar separation
    return r"\rowcolor{gray!25} RR(" + name + ")\n& " + " & ".join(out) + r" \\"

rows = {
    r"$\theta_{\text{mix}}{-}\tau_{r}$": rr(mix_tau, mix, clean),
    r"$\theta_{\text{SAP}}$": rr(sap, mix, clean),
    # (Optional) verify existing rows too
    r"$\theta_{\text{CF}}$": rr(cf, mix, clean),
    r"$\hat{\theta}^{\ast}_u$": rr(uhat, mix, clean),
}

for k, v in rows.items():
    print(latex_row(k, v))