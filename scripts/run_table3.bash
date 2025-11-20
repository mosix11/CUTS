# commands to run the experiments of Table 3 in the paper.
# results will be saved in results/clip_poison_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/clip_poison_TA

###############
# CIFAR10     #
# poison      #
###############

# poison ratio = 2%  -> config2
python run_experiment.py -e poison -a clip -c config2 -f -t -p;

# poison ratio = 10% -> config6
python run_experiment.py -e poison -a clip -c config6 -f -t -p;

# poison ratio = 20% -> config4
python run_experiment.py -e poison -a clip -c config4 -f -t -p;


###############
# CIFAR100    #
# poison      #
###############

# poison ratio = 2%  -> config3
python run_experiment.py -e poison -a clip -c config3 -f -t -p;

# poison ratio = 10% -> config7
python run_experiment.py -e poison -a clip -c config7 -f -t -p;

# poison ratio = 20% -> config5
python run_experiment.py -e poison -a clip -c config5 -f -t -p;
