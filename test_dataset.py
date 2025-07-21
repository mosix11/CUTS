
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic, data_utils

import torch
import torchvision
from pathlib import Path
import copy
batch_size = 1024
# subsample_size = (10, 1000)
img_size = (32,32)
class_subset = []
remap_labels = False
balance_classes = False
label_noise = 0.3


grayscale = False
flatten = False
normalize_imgs = True

training_seed = 11
dataset_seed = 11


# heldout_conf = {
#     0: 0.5,
#     1: 0.5,
#     2: 0.5,
#     3: 0.5,
#     4: 0.5,
#     5: 0.5,
#     6: 0.5,
#     7: 0.5,
#     8: 0.5,
#     9: 0.5
# }

orig_trainset = copy.deepcopy(torchvision.datasets.CIFAR10(root=Path('data/CIFAR10'), train=True, download=True))
orig_testset = copy.deepcopy(torchvision.datasets.CIFAR10(root=Path('data/CIFAR10'), train=False, download=True))

# orig_trainset = torchvision.datasets.CIFAR10(root=Path('data/CIFAR10'), train=True, download=True)
# orig_testset = torchvision.datasets.CIFAR10(root=Path('data/CIFAR10'), train=False, download=True)

dataset = CIFAR10(
        batch_size=batch_size,
        # subsample_size=subsample_size,
        img_size=img_size,
        grayscale=grayscale,
        flatten=flatten,
        class_subset=class_subset,
        remap_labels=remap_labels,
        balance_classes=balance_classes,
        # augmentations=augmentations,
        normalize_imgs=normalize_imgs,
        valset_ratio=0.0,
        
        num_workers=8,
        seed=dataset_seed,
    )


# dataset.inject_noise(
#     set='Train',
#     noise_rate=label_noise,
#     noise_type='symmetric',
# )

dataset.inject_noise(
    set='Train',
    noise_rate=label_noise,
    noise_type='asymmetric',
)


num_total_samples = len(orig_trainset)
num_clean_samples = int(num_total_samples * (1-label_noise))
num_noisy_samples = num_total_samples - num_clean_samples

trgts = []

num_unmatched_lbls = 0

my_trainset = dataset.get_trainset() 
for idx in range(len(orig_trainset)):
    _, orig_lbl = orig_trainset[idx]
    _, my_lbl, _, _ = my_trainset[idx]
    
    if orig_lbl != my_lbl:
        if orig_lbl == 9:
            print(f"9:{my_lbl}")
        elif orig_lbl == 2:
            print(f"2:{my_lbl}")
        elif orig_lbl == 3:
            print(f"3:{my_lbl}")
        elif orig_lbl == 5:
            print(f"5:{my_lbl}")
        elif orig_lbl == 4:
            print(f"4:{my_lbl}")
        num_unmatched_lbls += 1
    trgts.append(orig_lbl)
        
print(f"Out of {num_total_samples}, {num_noisy_samples} were noisy and {num_unmatched_lbls} were detected")

# trgts[-10:] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# dummy_instance = my_trainset
# while not isinstance(dummy_instance, data_utils.NoisyClassificationDataset):
#     dummy_instance = dummy_instance.dataset
# dummy_instance.replace_labels(trgts)

# for idx in range(len(my_trainset)):
#     _, my_lbl, _, _ = my_trainset[idx]
#     if my_lbl == -1:
#         print(idx)

dummy_instance = my_trainset
while not isinstance(dummy_instance, data_utils.NoisyClassificationDataset):
    dummy_instance = dummy_instance.dataset
dummy_instance.switch_to_clean_lables()

num_unmatched_lbls = 0

# my_trainset = dataset.get_trainset() 
for idx in range(len(orig_trainset)):
    _, orig_lbl = orig_trainset[idx]
    _, my_lbl, _, _ = my_trainset[idx]
    
    if orig_lbl != my_lbl:
        num_unmatched_lbls += 1
    

print(f"Out of {num_total_samples}, {num_noisy_samples} were noisy and {num_unmatched_lbls} were detected")










# train_dl = dataset.get_train_dataloader()


# labels = set()
# num_tot = 0
# num_noisy = 0
# for batch in train_dl:
    
#     if len(batch) == 3:
#         x, y, is_noisy = batch
#     else:
#         x, y = batch
#         is_noisy = torch.zeros_like(y, dtype=torch.bool)
#     # print(x.shape, y.shape, is_noisy.shape)
#     # print(y)
#     # print(is_noisy)
#     num_tot += x.shape[0]
#     num_noisy += torch.sum(is_noisy).item()
#     labels.update(y.tolist())
# print(labels)
# print(num_tot)
# print(num_noisy)

# print('\n\n\n')
# held_dl = dataset.get_heldout_dataloader()

# labels = set()
# num_tot = 0
# num_noisy = 0
# for batch in held_dl:
#     if len(batch) == 3:
#         x, y, is_noisy = batch
#     else:
#         x, y = batch
#         is_noisy = torch.zeros_like(y, dtype=torch.bool)
#     # print(x.shape, y.shape, is_noisy.shape)
#     # print(y)
#     # print(is_noisy)
#     num_tot += x.shape[0]
#     num_noisy += torch.sum(is_noisy).item()
#     labels.update(y.tolist())
# print(labels)
# print(num_tot)
# print(num_noisy)
