# commands to run the experiments of Table 10 in the paper.
# results will be saved in results/dino_poison_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/dino_poison_TA

###########
# CIFAR10 #
###########

# poison rate = 2%
python run_experiment.py -e poison -a dino -c config1 -f -t -p;

# poison rate = 10%
python run_experiment.py -e poison -a dino -c config3 -f -t -p;

# poison rate = 20%
python run_experiment.py -e poison -a dino -c config4 -f -t -p;


############
# CIFAR100 #
############

# poison rate = 2%
python run_experiment.py -e poison -a dino -c config2 -f -t -p;

# poison rate = 10%
python run_experiment.py -e poison -a dino -c config5 -f -t -p;

# poison rate = 20%
python run_experiment.py -e poison -a dino -c config6 -f -t -p;
