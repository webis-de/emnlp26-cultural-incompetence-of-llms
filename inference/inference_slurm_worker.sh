#Example body (inference slurm) 

source .venv/bin/activate

CONFIG="<REDACTED_PATH>/inference_config.json"

MODELS=($(jq -r '.models | keys[]' $CONFIG))
SEEDS=($(jq -r '.seeds[]' $CONFIG))
SPLITS=$(jq -r '.num_splits' $CONFIG)
CONCURRENT=$(jq -r '.max_concurrent_jobs' $CONFIG)

NUM_MODELS=${#MODELS[@]}
NUM_SEEDS=${#SEEDS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_SEEDS * SPLITS) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_SEEDS * SPLITS) ))

SEED_IDX=$(( REMAINDER / SPLITS ))
SPLIT_IDX=$(( REMAINDER % SPLITS ))

MODEL_NAME=${MODELS[$MODEL_IDX]}
CURRENT_SEED=${SEEDS[$SEED_IDX]}

MODEL_PATH=$(jq -r ".models.\"$MODEL_NAME\".path" $CONFIG)
QUANT=$(jq -r ".models.\"$MODEL_NAME\".quantization" $CONFIG)
LORA=$(jq -r ".models.\"$MODEL_NAME\".lora_path" $CONFIG)
DTYPE=$(jq -r ".models.\"$MODEL_NAME\".dtype" $CONFIG)

ARGS="--model_name $MODEL_NAME --model_path $MODEL_PATH --seed $CURRENT_SEED --split_id $SPLIT_IDX --num_splits $SPLITS --config $CONFIG"

[[ "$QUANT" != "null" ]] && ARGS="$ARGS --quantization $QUANT"
[[ "$LORA" != "null" ]] && ARGS="$ARGS --lora_path $LORA"
[[ "$DTYPE" != "null" ]] && ARGS="$ARGS --dtype $DTYPE"

python <REDACTED_PATH>/inference_worker_transformers.py $ARGS
