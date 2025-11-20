# commands to run the experiments of Figure 4 in the paper.
# results will be saved in results/regular_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# --- Symmetric noise (SN), Resnet18/34/50/101 Randomly Initialized ---

# ResNet18, eta = 40
python run_experiment.py -e noise -a regular -c config23 -f -t -s;
# ResNet34, eta = 40
python run_experiment.py -e noise -a regular -c config35 -f -t -s;
# ResNet50, eta = 40
python run_experiment.py -e noise -a regular -c config37 -f -t -s;
# ResNet101, eta = 40
python run_experiment.py -e noise -a regular -c config39 -f -t -s;


# --- Symmetric noise (SN), Resnet18/34/50/101 Initialized with ImageNet1Kv1 weights ---

# ResNet18, eta = 40
python run_experiment.py -e noise -a regular -c config34 -f -t -s;
# ResNet34, eta = 40
python run_experiment.py -e noise -a regular -c config36 -f -t -s;
# ResNet50, eta = 40
python run_experiment.py -e noise -a regular -c config38 -f -t -s;
# ResNet101, eta = 40
python run_experiment.py -e noise -a regular -c config40 -f -t -s;


# --- Symmetric noise (SN), Resnet18/34/50/101 Initialized with ImageNet1Kv2 weights ---

# ResNet50, eta = 40
python run_experiment.py -e noise -a regular -c config41 -f -t -s;
# ResNet101, eta = 40
python run_experiment.py -e noise -a regular -c config42 -f -t -s;