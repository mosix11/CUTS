# commands to run the experiments of Table 2 in the paper.
# results will be saved in results/regular_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/regular_noise_TA

###########
# MNIST   #
# FC1     #
###########

# SN 20%  -> config17
python run_experiment.py -e noise -a regular -c config17 -f -t -s;

# SN 40%  -> config18
python run_experiment.py -e noise -a regular -c config18 -f -t -s;

# AN 40%  -> config31
python run_experiment.py -e noise -a regular -c config31 -f -t -s;


###############
# CIFAR10     #
# ResNet18    #
###############

# SN 20%  -> config22
python run_experiment.py -e noise -a regular -c config22 -f -t -s;

# SN 40%  -> config23
python run_experiment.py -e noise -a regular -c config23 -f -t -s;

# AN 40%  -> config32
python run_experiment.py -e noise -a regular -c config32 -f -t -s;


###############
# CIFAR100    #
# ResNet18    #
###############

# SN 20%  -> config27
python run_experiment.py -e noise -a regular -c config27 -f -t -s;

# SN 40%  -> config28
python run_experiment.py -e noise -a regular -c config28 -f -t -s;

# AN 40%  -> config33
python run_experiment.py -e noise -a regular -c config33 -f -t -s;
