
for x in {34..41}
do
    echo "Running experiment with config$x"
    python regular_noise_experiments.py -c config$x -t
done