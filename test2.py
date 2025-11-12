mix  = [98.7,96.0,81.7,57.1,24.8, 96.9,93.5,76.2,51.9,22.7, 94.7,83.1, 88.2,84.6,72.5,52.1,24.9, 80.9,60.5]
clean= [99.7,99.7,99.7,99.6,99.5, 98.6,98.7,98.5,98.4,98.0, 98.7,98.4, 90.2,90.2,89.8,88.9,86.9, 90.2,89.4]

mix_tau = [98.7,96.1,81.8,57.1,25.2, 96.8,93.5,76.2,52.0,22.7, 94.7,83.1, 88.2,84.5,72.4,52.0,24.7, 80.9,60.6]
sap     = [99.3,98.6,95.0,92.0,58.4, 97.0,95.1,85.0,72.0,26.3, 95.2,81.4, 88.1,85.0,75.8,59.4,28.5, 81.3,61.5]

def rr(model):
    out=[]
    for m,x,c in zip(model,mix,clean):
        out.append( round( (m-x)/(c-x)*100, 1 ) )
    return out

def fmt_row(name, vals):
    parts = " & ".join(f"{v:.1f}" for v in vals)
    return rf"\rowcolor{{gray!25}} RR({name})" + "\n& " + parts + r" \\"

print(fmt_row(r"$\theta_{\text{mix}}{-}\tau_{r}$", rr(mix_tau)))
print()
print(fmt_row(r"$\theta_{\text{SAP}}$", rr(sap)))
