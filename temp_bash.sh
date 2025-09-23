
for x in {21..30}
do
    echo "Running experiment with config$x"
    CUDA_VISIBLE_DEVICES=1 python regular_noise_experiments.py -c config$x -f
done