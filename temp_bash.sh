
for x in {30..33}
do
    echo "Running experiment with config$x"
    python regular_noise_experiments.py -c config$x -f
done