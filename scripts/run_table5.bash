# commands to run the experiments of Table 5 in the paper.
# results will be saved in results/regular_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/regular_noise_TA


###############
# MNIST + FC1 #
###############

# --- Symmetric noise (SN) ---
# eta = 10
python run_experiment.py -e noise -a regular -c config16 -f -t -s;
# eta = 20
python run_experiment.py -e noise -a regular -c config17 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config18 -f -t -s;
# eta = 60
python run_experiment.py -e noise -a regular -c config19 -f -t -s;
# eta = 80
python run_experiment.py -e noise -a regular -c config20 -f -t -s;

# --- Asymmetric noise (AN) ---
# eta = 20
python run_experiment.py -e noise -a regular -c config45 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config31 -f -t -s;


########################
# CIFAR10 + ResNet18    #
########################

# --- Symmetric noise (SN), ResNet18 ---
# eta = 10
python run_experiment.py -e noise -a regular -c config21 -f -t -s;
# eta = 20
python run_experiment.py -e noise -a regular -c config22 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config23 -f -t -s;
# eta = 60
python run_experiment.py -e noise -a regular -c config24 -f -t -s;
# eta = 80
python run_experiment.py -e noise -a regular -c config25 -f -t -s;


# --- Asymmetric noise (AN), ResNet18 ---
# eta = 20
python run_experiment.py -e noise -a regular -c config44 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config32 -f -t -s;


#########################
# CIFAR100 + ResNet18   #
#########################

# --- Symmetric noise (SN) ---
# eta = 10
python run_experiment.py -e noise -a regular -c config26 -f -t -s;
# eta = 20
python run_experiment.py -e noise -a regular -c config27 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config28 -f -t -s;
# eta = 60
python run_experiment.py -e noise -a regular -c config29 -f -t -s;
# eta = 80
python run_experiment.py -e noise -a regular -c config30 -f -t -s;

# --- Asymmetric noise (AN) ---
# eta = 20
python run_experiment.py -e noise -a regular -c config43 -f -t -s;
# eta = 40
python run_experiment.py -e noise -a regular -c config33 -f -t -s;
