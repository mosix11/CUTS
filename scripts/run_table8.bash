# commands to run the experiments of Table 8 in the paper.
# results will be saved in results/regular_poison_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/regular_poison_TA

############
# MNIST    #
# R18      #
############
# 2% poison -> config4
python run_experiment.py -e poison -a regular -c config4 -f -t -p;

# 10% poison -> config6
python run_experiment.py -e poison -a regular -c config6 -f -t -p;

# 20% poison -> config7
python run_experiment.py -e poison -a regular -c config7 -f -t -p;


###############
# CIFAR10     #
# R18         #
###############
# 2% poison -> config2
python run_experiment.py -e poison -a regular -c config2 -f -t -p;

# 10% poison -> config8
python run_experiment.py -e poison -a regular -c config8 -f -t -p;

# 20% poison -> config9
python run_experiment.py -e poison -a regular -c config9 -f -t -p;


###############
# CIFAR100    #
# R18         #
###############
# 2% poison -> config3
python run_experiment.py -e poison -a regular -c config3 -f -t -p;

# 10% poison -> config10
python run_experiment.py -e poison -a regular -c config10 -f -t -p;

# 20% poison -> config11
python run_experiment.py -e poison -a regular -c config11 -f -t -p;

