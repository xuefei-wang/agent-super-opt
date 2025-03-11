#!/bin/sh

OUTPUT_FOLDER="output/"

echo "Initializing output folder"

mkdir $OUTPUT_FOLDER
cd $OUTPUT_FOLDER
touch preprocessing_func_bank.json
echo "[]" > preprocessing_func_bank.json
cd ..

echo "Starting agent pipeline"

python main_agent.py --output $OUTPUT_FOLDER

