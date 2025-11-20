# commands to run the experiments of Table 1 in the paper.
# results will be saved in results/clip_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/clip_noise_TA

############################
# MNIST - symmetric #
############################

# eta = 10
python run_experiment.py -e noise -a clip -c config37 -f -t -s;

# eta = 20
python run_experiment.py -e noise -a clip -c config38 -f -t -s;

# eta = 40
python run_experiment.py -e noise -a clip -c config39 -f -t -s;

# eta = 60
python run_experiment.py -e noise -a clip -c config40 -f -t -s;

# eta = 80
python run_experiment.py -e noise -a clip -c config41 -f -t -s;


#############################
# CIFAR10 - symmetric #
#############################

# eta = 10
python run_experiment.py -e noise -a clip -c config32 -f -t -s;

# eta = 20
python run_experiment.py -e noise -a clip -c config30 -f -t -s;

# eta = 40
python run_experiment.py -e noise -a clip -c config26 -f -t -s;

# eta = 60
python run_experiment.py -e noise -a clip -c config28 -f -t -s;

# eta = 80
python run_experiment.py -e noise -a clip -c config35 -f -t -s;


################################
# CIFAR10 - asymmetric  #
################################

# eta = 20
python run_experiment.py -e noise -a clip -c config44 -f -t -s;

# eta = 40
python run_experiment.py -e noise -a clip -c config42 -f -t -s;


##############################
# CIFAR100 - symmetric #
##############################

# eta = 10
python run_experiment.py -e noise -a clip -c config34 -f -t -s;

# eta = 20
python run_experiment.py -e noise -a clip -c config31 -f -t -s;

# eta = 40
python run_experiment.py -e noise -a clip -c config27 -f -t -s;

# eta = 60
python run_experiment.py -e noise -a clip -c config29 -f -t -s;

# eta = 80
python run_experiment.py -e noise -a clip -c config36 -f -t -s;


#################################
# CIFAR100 - asymmetric  #
#################################

# eta = 20
python run_experiment.py -e noise -a clip -c config45 -f -t -s;

# eta = 40
python run_experiment.py -e noise -a clip -c config43 -f -t -s;
