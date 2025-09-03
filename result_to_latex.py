import argparse
import os
import dotenv
import yaml
import json
import re
from pathlib import Path

from collections import OrderedDict


if __name__ == "__main__":
    with open('configmap.yaml', 'r') as file:
        configmap = yaml.full_load(file)
    
    clip_models_cfgs = configmap['clip_models']
    clip_models_results_dir = Path('results/single_experiment/clip_noise_TA')
    
    clip_symmetric_cfgs = OrderedDict()
    clip_symmetric_cfgs['MNIST'] = clip_models_cfgs['MNIST']['symmetric']
    clip_symmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['symmetric']
    clip_symmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['symmetric']
    
    
    clip_asymmetric_cfgs = OrderedDict()
    clip_asymmetric_cfgs['MNIST'] = clip_models_cfgs['MNIST']['asymmetric']
    clip_asymmetric_cfgs['CIFAR10'] = clip_models_cfgs['CIFAR10']['asymmetric']
    clip_asymmetric_cfgs['CIFAR100'] = clip_models_cfgs['CIFAR100']['asymmetric']
    
    print(clip_symmetric_cfgs)
    
    regular_models_cfgs = configmap['regular_models']
    clip_models_results_dir = Path('results/single_experiment/pretrain_on_noisy')
    
    
    