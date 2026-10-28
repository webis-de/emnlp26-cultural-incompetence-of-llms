#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export TRITON_CACHE_DIR="/tmp/msplith_triton_cache"

# Check if a dataset name is provided as the first argument.
if [ -z "$1" ]; then
  echo "Error: No dataset name provided." >&2
  echo "Usage: $0 [ultrachat-400k-nl|all-nl|stackoverflow-nl|norobots-nl|ultrachat-200k-en|norobots-en|stackoverflow-en|belle-05m-zh]" >&2
  exit 1
fi

DATASET_NAME=$1

# Determine which configuration file to use based on the dataset name.
case "$DATASET_NAME" in
  ultrachat-400k-nl)
    CONFIG_FILE="qwen_config_qlora_ultrachat-400k-nl.yaml"
    PORT=29051
    ;;
  all-nl)
    CONFIG_FILE="qwen_config_qlora_all-nl.yaml"
    PORT=29052
    ;;
  stackoverflow-nl)
    CONFIG_FILE="qwen_config_qlora_stackoverflow-nl.yaml"
    PORT=29053
    ;;
  norobots-nl)
    CONFIG_FILE="qwen_config_qlora_norobots-nl.yaml"
    PORT=29054
    ;;
  ultrachat-200k-en)
    CONFIG_FILE="qwen_config_qlora_ultrachat-200k-en.yaml"
    PORT=29055
    ;;
  norobots-en)
    CONFIG_FILE="qwen_config_qlora_norobots-en.yaml"
    PORT=29056
    ;;
  stackoverflow-en)
    CONFIG_FILE="qwen_config_qlora_stackoverflow-en.yaml"
    PORT=29057
    ;;
  belle-05m-zh)
    CONFIG_FILE="qwen_config_qlora_belle-0.5m-zh.yaml"
    PORT=29058
    ;;
  belle-05m-zh-full)
    CONFIG_FILE="qwen_config_lora_belle-0.5m-zh.yaml"
    PORT=29059
    ;;
  *)
    echo "Error: Invalid dataset name '$DATASET_NAME'." >&2
    exit 1
    ;;
esac

echo "Starting SFT for dataset: $DATASET_NAME with config: $CONFIG_FILE"

accelerate launch \
    --config_file recipes/accelerate_configs/ddp.yaml \
    --num_processes=2 \
    --main_process_port "$PORT" \
    scripts/sft.py \
        --config "$CONFIG_FILE"
