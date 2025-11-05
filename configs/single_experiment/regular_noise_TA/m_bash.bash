#!/bin/bash

# Loop through all YAML config files
for file in config*.yaml; do
  echo "Processing $file ..."

  # Create a backup
  cp "$file" "$file.bak"

  # Perform in-place key renaming inside strategy block
  sed -i '
    s/\bfinetuning_set:/proxy_set:/g;
    s/\&finetuning_set/&proxy_set/g;
    s/\bnoise:/corruption:/g;
    s/\bpretraining:/mix:/g;
    s/\bfinetuning:/proxy:/g
  ' "$file"

done

echo "✅ All files processed. Backup copies (*.bak) are kept for safety."
