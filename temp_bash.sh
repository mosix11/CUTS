
for x in {16..30}
do
    echo "Running experiment with config$x"
    python regular_noise_experiments.py -c config$x -f
done