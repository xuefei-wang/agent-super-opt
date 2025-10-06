#!/bin/bash

# Define the list of JSON file paths (excluding those with "latest" in their names)
INITIAL_PATHS=(/data/xwang3/Projects/agent-medsam/cellpose_segmentation/*/preprocessing_func_bank.json)

JSON_PATHS=()
for path in "${INITIAL_PATHS[@]}"; do
    # Check if the path does NOT contain "latest"
    if [[ ! "$path" == *"latest"* ]]; then
        # If it doesn't, add it to our final array
        JSON_PATHS+=("$path")
    fi
done

# Define the available GPU IDs
GPUS=(2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}
JOB_COUNTER=0

# Loop through each JSON path and distribute the command
for JSON_PATH in "${JSON_PATHS[@]}"; do
    # Assign a GPU in a round-robin fashion
    GPU_ID=${GPUS[$((JOB_COUNTER % NUM_GPUS))]}
    
    echo "Running job on GPU $GPU_ID with path: $JSON_PATH"
    
    # Run the command in the background
    python figs/cellpose_analyze_trajectories.py --json_path="$JSON_PATH" --gpu_id="$GPU_ID" --data_path="/data/xwang3/sci-agent-data/cellpose_data/updated_cellpose_combined_data/" &
    
    # Increment the job counter
    ((JOB_COUNTER++))
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed. ✅"