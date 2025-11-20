# commands to run the experiments of Table 7 in the paper.
# results will be saved in results/dino_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail


cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/dino_noise_TA

###############
# MNIST       #
###############

# MNIST, SN, eta=40 -> config3
python run_experiment.py -e noise -a dino -c config3 -f -t -s;


################
# CIFAR10      #
################

# CIFAR10, SN, eta=40 -> config1
python run_experiment.py -e noise -a dino -c config1 -f -t -s;

# CIFAR10, AN, eta=40 -> config4
python run_experiment.py -e noise -a dino -c config4 -f -t -s;


################
# CIFAR100     #
################

# CIFAR100, SN, eta=40 -> config2
python run_experiment.py -e noise -a dino -c config2 -f -t -s;

# CIFAR100, AN, eta=40 -> config5
python run_experiment.py -e noise -a dino -c config5 -f -t -s;
