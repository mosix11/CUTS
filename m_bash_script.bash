for i in {26..45}; do
  if [ "$i" -eq 33 ]; then
    echo "Skipping config33"
    continue
  fi
  echo "Running config$i..."
  CUDA_VISIBLE_DEVICES=1 python run_experiment.py -e noise -a clip -c config$i -s
done