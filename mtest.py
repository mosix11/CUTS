from src.datasets import CIFAR10, MNIST, CIFAR100
import torch
import matplotlib.pyplot as plt


def show_poisoned_samples(dataset, n=9, unnormalize=False):
    """
    Show `n` poisoned samples from a DatasetWithIndex-wrapped dataset.
    dataset: your ds.get_trainset() or similar
    unnormalize: if True, try to undo CIFAR-10 normalization for visualization
    """

    # Collect poisoned samples
    poisoned_imgs = []
    poisoned_labels = []
    for idx in range(len(dataset)):
        x, y, *_rest = dataset[idx]
        # last element in your tuple is is_poison flag
        is_poison = _rest[-1].item() if torch.is_tensor(_rest[-1]) else bool(_rest[-1])
        if is_poison:
            poisoned_imgs.append(x)
            poisoned_labels.append(y)
            if len(poisoned_imgs) >= n:
                break

    if not poisoned_imgs:
        print("No poisoned samples found!")
        return

    # CIFAR-10 normalization parameters
    cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    cifar_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

    # Plot grid
    fig, axes = plt.subplots(3, 3, figsize=(8,8))
    for ax, img, label in zip(axes.flat, poisoned_imgs, poisoned_labels):
        if unnormalize:
            img = img * cifar_std + cifar_mean
        img = img.clamp(0,1)
        if img.shape[0] == 1:  # grayscale
            ax.imshow(img.squeeze(0).cpu(), cmap="gray")
        else:
            ax.imshow(img.permute(1,2,0).cpu())
        ax.set_title(f"Label={label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
# ds = CIFAR10(normalize_imgs=True, seed=42)
# ds.inject_poison(set='Train', rate=0.02, target_class=0, margin=(10,6), seed=70)
ds = MNIST(normalize_imgs=True, seed=42)
ds.inject_poison(set='Train', rate=0.02, target_class=0, margin=3, seed=99)

show_poisoned_samples(ds.get_trainset(), n=9, unnormalize=False)

# for idx, sample in enumerate(ds.get_trainset()):
#     x, y, dx, is_poison = sample
    
#     # x_norm, y_norm, dx_nrom, is_poison_norm = ds_norm.get_trainset()[idx]
    
#     if is_poison:
#         percentage = (x == x.max()).sum().item() * 100 / x.numel()
#         print(percentage)
        
        