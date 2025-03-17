#!/bin/sh

OUTPUT_FOLDER="output2/"

DATASET="spot_data/SpotNet-v1_1/val.npz"

echo "Initializing output folder"

mkdir $OUTPUT_FOLDER
cd $OUTPUT_FOLDER
touch preprocessing_func_bank.json
echo "[]" > preprocessing_func_bank.json
cd ..

echo "Starting agent pipeline"

python main_agent.py --dataset $DATASET --output $OUTPUT_FOLDER

