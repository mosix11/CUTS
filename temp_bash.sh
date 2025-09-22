
for x in {9..15}
do
    echo "Running experiment with config$x"
    CUDA_VISIBLE_DEVICES=0 python regular_noise_experiments.py -c config$x -f
done