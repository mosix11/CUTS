# commands to run the experiments of Table 11 in the paper.
# results will be saved in results/clip_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail


cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/clip_noise_TA

###############
# MNIST SN    #
###############

# 2% Proxy Size
# MNIST, eta=40 
python run_experiment.py -e noise -a clip -c config39 -f -t -s;
# MNIST, eta=60 
python run_experiment.py -e noise -a clip -c config40 -f -t -s;

# 10% Proxy Size
# MNIST, eta=40 
python run_experiment.py -e noise -a clip -c config3 -f -t -s;
# MNIST, eta=60 
python run_experiment.py -e noise -a clip -c config23 -f -t -s;



################
# CIFAR10 SN   #
################


# 2% Proxy Size
# CIFAR10,  eta=40 
python run_experiment.py -e noise -a clip -c config26 -f -t -s;
# CIFAR10,  eta=60 
python run_experiment.py -e noise -a clip -c config28 -f -t -s;

# 10% Proxy Size
# CIFAR10,  eta=40 
python run_experiment.py -e noise -a clip -c config1 -f -t -s;
# CIFAR10,  eta=60 
python run_experiment.py -e noise -a clip -c config11 -f -t -s;


################
# CIFAR100 SN  #
################

# 2% Proxy Size
# CIFAR100,  eta=40 
python run_experiment.py -e noise -a clip -c config27 -f -t -s;
# CIFAR100,  eta=60 
python run_experiment.py -e noise -a clip -c config29 -f -t -s;

# 10% Proxy Size
# CIFAR100,  eta=40 
python run_experiment.py -e noise -a clip -c config2 -f -t -s;
# CIFAR100,  eta=60 
python run_experiment.py -e noise -a clip -c config12 -f -t -s;
