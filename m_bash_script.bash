for i in {21..42}; do
  echo "Running config$i..."
  CUDA_VISIBLE_DEVICES=1 python run_experiment.py -e noise -a regular -c config$i -s
done