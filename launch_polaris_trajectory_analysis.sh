#!/bin/bash

# --- Configuration ---
# Define the available GPU IDs
GPUS=(2 3 4 6)

# --- Find JSON files ---
# Find all relevant JSON files, excluding any with "latest" in the path.
INITIAL_PATHS=(spot_detection/*/preprocessing_func_bank.json)
JSON_PATHS=()
for path in "${INITIAL_PATHS[@]}"; do
    if [[ ! "$path" == *"latest"* ]]; then
        JSON_PATHS+=("$path")
    fi
done

# --- Automatically determine counts ---
NUM_GPUS=${#GPUS[@]}
NUM_JOBS=${#JSON_PATHS[@]}

echo "Found ${NUM_JOBS} JSON files to process."
echo "Configured to use ${NUM_GPUS} GPUs: ${GPUS[*]}"

# --- Define the Worker Function ---
# This function runs on a single GPU and processes all jobs assigned to it, one by one.
# It takes two arguments: the GPU_ID to use and the worker's index (e.g., 0, 1, 2...).
run_jobs_on_gpu() {
  local GPU_ID=$1
  local WORKER_INDEX=$2 

  echo "Worker for GPU ${GPU_ID} (Index ${WORKER_INDEX}) has started. PID: $$"

  # Loop through all job indices from 0 to NUM_JOBS-1
  for i in "${!JSON_PATHS[@]}"; do
    # Assign a job to this worker if the job index modulo the number of GPUs
    # matches this worker's index. This distributes jobs evenly.
    if (( i % NUM_GPUS == WORKER_INDEX )); then
      JSON_PATH=${JSON_PATHS[$i]}
      
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID} starting job $((i+1))/${NUM_JOBS}: ${JSON_PATH}"
      
      # Run the python command in the foreground (within this backgrounded function).
      # This is the key change: the worker waits for this command to finish
      # before starting its next assigned job.
      python figs/spot_detection_analyze_trajectories.py  --json_path="$JSON_PATH" --data_path='/data/xwang3/sci-agent-data/SpotNet-v1_1'
      
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID} finished job: ${JSON_PATH}"
      echo "-----------------------------------------------------"
    fi
  done
  echo "Worker for GPU ${GPU_ID} has completed all its assigned jobs."
}

# --- Main script execution ---

# Export the function and variables so they are available to the subshells
# created by running the workers in the background.
export -f run_jobs_on_gpu
export JSON_PATHS NUM_GPUS

echo "Launching parallel workers..."

# Launch one worker function for each GPU in the background
for index in "${!GPUS[@]}"; do
  GPU_ID=${GPUS[$index]}
  # The '&' runs the entire 'run_jobs_on_gpu' function in the background.
  run_jobs_on_gpu ${GPU_ID} ${index} &
done

echo "All workers have been launched. Waiting for them to complete..."
# The 'wait' command waits for all backgrounded child processes (the workers) to finish.
wait
echo "All jobs have completed. ✅"