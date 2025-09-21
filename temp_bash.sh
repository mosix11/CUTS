
for x in {34..45}
do
    echo "Running experiment with config$x"
    python clip_noisy_experiments.py -c config$x -t
done