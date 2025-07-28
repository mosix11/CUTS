from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic, data_utils

import comet_ml
from src.datasets import dataset_factory, data_utils
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
import torch
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, Subset, ConcatDataset
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv
import yaml
import pickle
import copy
import random
import numpy as np
from torchmetrics import ConfusionMatrix
import json
from tqdm import tqdm
from collections import OrderedDict





def test(cfg):
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    print(len(dataset.get_trainset()))
    print(len(dataset.get_testset()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )


    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/pretrain_on_noisy') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    test(cfg)