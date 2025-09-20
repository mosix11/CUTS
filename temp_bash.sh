
for x in {26..45}
do
    echo "Running experiment with config$x"
    CUDA_VISIBLE_DEVICES=1 python clip_noisy_experiments.py -c config$x -t
done