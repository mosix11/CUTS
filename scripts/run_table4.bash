# commands to run the experiments of Table 4 in the paper.
# results will be saved in results/dino_realworld_TA/configX

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."


# confgis from configs/dino_realworld_TA

python run_experiment.py -e noise -a dino -r -c config5 -f -t;

# transfer the pre-trained weights from the previous run, so we don't train the same mix model again.
mkdir outputs/dino_realworld_TA/config6; cp -r outputs/dino_realworld_TA/config5/mix outputs/dino_realworld_TA/config6/;
mkdir outputs/dino_realworld_TA/config7; cp -r outputs/dino_realworld_TA/config5/mix outputs/dino_realworld_TA/config7/;
mkdir outputs/dino_realworld_TA/config8; cp -r outputs/dino_realworld_TA/config5/mix outputs/dino_realworld_TA/config8/;

python run_experiment.py -e noise -a dino -r -c config6 -f -t;

python run_experiment.py -e noise -a dino -r -c config7 -f -t;

python run_experiment.py -e noise -a dino -r -c config8 -f -t;

# Final result shown in the table is the average of these four runs.