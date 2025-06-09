import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from src.datasets.utils import apply_label_noise, NoisyDataset


def subset_low_loss_samples(model, dataset):
    original_training_set = dataset.get_trainset()
    train_dl =  DataLoader(
            original_training_set,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    
    