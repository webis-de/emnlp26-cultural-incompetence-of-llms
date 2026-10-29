#Example body (inference slurm) 

source .venv/bin/activate

CONFIG="<REDACTED_PATH>/inference_config.json"

#compute variables (number of tasks)
NUM_MODELS=$(jq '.models | length' $CONFIG)
NUM_SEEDS=$(jq '.seeds | length' $CONFIG)
NUM_SPLITS=$(jq '.num_splits' $CONFIG)
CONCURRENT=$(jq '.max_concurrent_jobs' $CONFIG)

TOTAL_TASKS=$(( NUM_MODELS * NUM_SEEDS * NUM_SPLITS ))
ARRAY_LIMIT=$(( TOTAL_TASKS - 1 ))

echo "Launching Experiment Matrix:"
echo "- Models: $NUM_MODELS | Seeds: $NUM_SEEDS | Splits: $NUM_SPLITS"
echo "- Total Slurm Tasks: $TOTAL_TASKS"

#start the array
sbatch --array=0-${ARRAY_LIMIT}%${CONCURRENT} <REDACTED_PATH>/inference_slurm_worker.sh
