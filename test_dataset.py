
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic, data_utils
import torch

batch_size = 1024
subsample_size = (10, 1000)
img_size = (16,16)
class_subset = []
remap_labels = False
balance_classes = True
label_noise = 0.3


grayscale = True
flatten = True
normalize_imgs = False

training_seed = 11
dataset_seed = 11

# heldout_conf = (0.7, True)
# heldout_conf = {
#     0: (0.2, False),
#     1: (0.2, False),
#     2: (0.8, False),
#     3: (0.2, False),
#     4: (0.2, False),
#     5: (0.8, False),
#     6: (0.2, False),
#     7: (0.2, False),
#     8: (0.2, False),
#     9: (0.2, False),
# }
heldout_conf = {
    0: 0.5,
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    5: 0.5,
    6: 0.5,
    7: 0.5,
    8: 0.5,
    9: 0.5
}

dataset = CIFAR10(
        batch_size=batch_size,
        # subsample_size=subsample_size,
        img_size=img_size,
        grayscale=grayscale,
        flatten=flatten,
        class_subset=class_subset,
        remap_labels=remap_labels,
        balance_classes=balance_classes,
        heldout_conf=heldout_conf,
        # augmentations=augmentations,
        normalize_imgs=normalize_imgs,
        valset_ratio=0.0,
        
        num_workers=8,
        seed=dataset_seed,
    )


dataset.inject_noise(
    set='Train',
    noise_rate=label_noise,
    noise_type='symmetric',
)
dataset.inject_noise(
    set='Heldout',
    noise_rate=1.0,
    noise_type='symmetric',
)
train_dl = dataset.get_train_dataloader()


labels = set()
num_tot = 0
num_noisy = 0
for batch in train_dl:
    
    if len(batch) == 3:
        x, y, is_noisy = batch
    else:
        x, y = batch
        is_noisy = torch.zeros_like(y, dtype=torch.bool)
    # print(x.shape, y.shape, is_noisy.shape)
    # print(y)
    # print(is_noisy)
    num_tot += x.shape[0]
    num_noisy += torch.sum(is_noisy).item()
    labels.update(y.tolist())
print(labels)
print(num_tot)
print(num_noisy)

print('\n\n\n')
held_dl = dataset.get_heldout_dataloader()

labels = set()
num_tot = 0
num_noisy = 0
for batch in held_dl:
    if len(batch) == 3:
        x, y, is_noisy = batch
    else:
        x, y = batch
        is_noisy = torch.zeros_like(y, dtype=torch.bool)
    # print(x.shape, y.shape, is_noisy.shape)
    # print(y)
    # print(is_noisy)
    num_tot += x.shape[0]
    num_noisy += torch.sum(is_noisy).item()
    labels.update(y.tolist())
print(labels)
print(num_tot)
print(num_noisy)
