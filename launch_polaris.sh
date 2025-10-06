#!/bin/bash

# --- Configuration ---
# MODIFICATION: You can now change this array to any set of GPU IDs.
GPUS=(3 4 5 6) 
TOTAL_JOBS=20
DELAY_SECONDS=30 # Delay between launching the parallel workers

# --- Automatically determine the number of GPUs ---
NUM_GPUS=${#GPUS[@]}

# --- Define the base command to avoid repetition ---
BASE_CMD="python main.py \
  --dataset \"/data/xwang3/sci-agent-data/SpotNet-v1_1/val.npz\" \
  --experiment_name \"spot_detection\" \
  --history_threshold 5 \
  --k 3 \
  --k_word \"three\" "

# --- Define the worker function ---
# MODIFICATION: Takes two arguments: GPU_ID and its index in the array.
run_jobs_on_gpu() {
  GPU_ID=$1
  WORKER_INDEX=$2 # The index (0, 1, 2, ...) of this worker
  echo "Worker for GPU ${GPU_ID} (Index ${WORKER_INDEX}) has started. PID: $$"

  # Loop through all job numbers from 1 to TOTAL_JOBS
  for i in $(seq 1 ${TOTAL_JOBS}); do
    # MODIFICATION: Assign job based on its index, not the GPU ID
    ASSIGNED_INDEX=$(( (i - 1) % NUM_GPUS ))

    if [[ ${ASSIGNED_INDEX} -eq ${WORKER_INDEX} ]]; then
      SEED=$(( i + 100 ))
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Job #${i} on GPU ${GPU_ID} with seed ${SEED}..."
      
      # The actual GPU_ID is used here in the final command
      FULL_CMD="${BASE_CMD} --gpu_id ${GPU_ID} --random_seed ${SEED}"
      
      eval ${FULL_CMD}
      
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Job #${i} on GPU ${GPU_ID}."
      echo "-----------------------------------------------------"
    fi
  done
  echo "Worker for GPU ${GPU_ID} has completed all its jobs."
}

# --- Main script execution ---

export -f run_jobs_on_gpu
# MODIFICATION: Needed for the seq command to work inside the function
export TOTAL_JOBS NUM_GPUS BASE_CMD

echo "Launching parallel workers for ${NUM_GPUS} GPUs: ${GPUS[*]}..."

# MODIFICATION: Loop with an index to pass to the worker function
for index in "${!GPUS[@]}"; do
  GPU_ID=${GPUS[$index]}
  echo "-> Launching worker for GPU ${GPU_ID} (Index: ${index}) in the background."
  run_jobs_on_gpu ${GPU_ID} ${index} &
  
  echo "   Waiting ${DELAY_SECONDS}s before launching the next worker..."
  sleep ${DELAY_SECONDS}
done

echo "All workers have been launched. Waiting for them to complete..."
wait
echo "All jobs are done."