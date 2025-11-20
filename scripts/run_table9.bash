# commands to run the experiments of Table 9 in the paper.
# results will be saved in results/regular_poison_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/regular_poison_TA

###############
# CIFAR10     #
# vit         #
###############

# poison rate = 2%  -> config1
python run_experiment.py -e poison -a regular -c config1  -f -t -p;

# poison rate = 10% -> config12
python run_experiment.py -e poison -a regular -c config12 -f -t -p;

# poison rate = 20% -> config13
python run_experiment.py -e poison -a regular -c config13 -f -t -p;


###############
# CIFAR100    #
# vit         #
###############

# poison rate = 2%  -> config5
python run_experiment.py -e poison -a regular -c config5  -f -t -p;

# poison rate = 10% -> config14
python run_experiment.py -e poison -a regular -c config14 -f -t -p;

# poison rate = 20% -> config15
python run_experiment.py -e poison -a regular -c config15 -f -t -p;
