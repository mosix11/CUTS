# commands to run the experiments of Table 6 in the paper.
# results will be saved in results/regular_noise_TA/configX

#!/usr/bin/env bash
set -euo pipefail


cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/regular_noise_TA

#########################
# CIFAR10 + ViT   #
#########################

# --- Symmetric noise (SN) ---
# eta = 10 
python run_experiment.py -e noise -a regular -c config13 -f -t -s;

# eta = 20  
python run_experiment.py -e noise -a regular -c config1 -f -t -s;

# eta = 40  
python run_experiment.py -e noise -a regular -c config2 -f -t -s;

# eta = 60  
python run_experiment.py -e noise -a regular -c config5 -f -t -s;

# eta = 80  
python run_experiment.py -e noise -a regular -c config11 -f -t -s;

# --- Asymmetric noise (AN) ---
# eta = 20  
python run_experiment.py -e noise -a regular -c config7 -f -t -s;

# eta = 40  
python run_experiment.py -e noise -a regular -c config8 -f -t -s;


###########################
# CIFAR100 + ViT    #
###########################

# --- Symmetric noise (SN) ---
# eta = 10  
python run_experiment.py -e noise -a regular -c config14 -f -t -s;

# eta = 20  
python run_experiment.py -e noise -a regular -c config3 -f -t -s;

# eta = 40  
python run_experiment.py -e noise -a regular -c config4 -f -t -s;

# eta = 60  
python run_experiment.py -e noise -a regular -c config6 -f -t -s;

# eta = 80  
python run_experiment.py -e noise -a regular -c config12 -f -t -s;

# --- Asymmetric noise (AN) ---
# eta = 20  
python run_experiment.py -e noise -a regular -c config9 -f -t -s;

# eta = 40  
python run_experiment.py -e noise -a regular -c config10 -f -t -s;
