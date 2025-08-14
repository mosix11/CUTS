CONFIGS=(config70 config71 config72 config73 config74)   # your configs
MAX_JOBS=5                          # how many concurrent runs

mkdir -p logs

for cfg in "${CONFIGS[@]}"; do
  # throttle concurrency
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done

  CUDA_VISIBLE_DEVICES=0 \
  stdbuf -oL -eL python pretrain_on_noisy.py -c "$cfg" \
    > "logs/${cfg}.out" 2> "logs/${cfg}.err" &
  echo "Launched $cfg on GPU 0 (PID $!)"
done

wait
echo "All runs done."