# Agentic Superoptimization of Scientific Analysis Workflows

## Setup (For fully local execution)

We will use a virtual environment for this project. There are two requirements files - one shared and one task-dependent. 
You should add your task packages in `requirements_specific_{task_name}.txt` to run the following commands to set up the environment.

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Merge two requirement files into one
rm -f requirements.txt
cat requirements_shared.txt requirements_specific_{task_name}.txt > requirements.txt

# Install packages
pip install -r requirements.txt

# For MedSAM, you'll need to additionally run the following:
git clone https://github.com/sstilz/MedSAM.git
cd MedSAM
pip install -e .
```

## Getting Started

All commands should be executed from the project's root directory.
You data should be located outside the repo. The data path will be provided when initializing the task prompts.
You can find the attribute `dataset_path` under `TaskPrompts`.

0. Implement your tool
   
   Create a file under `src/`, wrap your tool into a class and provide comprehensive docstrings. Your tool should have `__init__`, `evaluate()`, `predict()` methods.

1. Test that your tool works as expected
   
   Basic version of tests is a manual pipeline testing that your tools works as expected. Additional unit tests are at your discretion. You can create a file under `tests/` folder and run the following command:
   
    ```bash
    python -m tests.test_X
    ```

2. Create your task specific prompts class with skeletonization

   Create a file under `prompts/` folder and wrap your task specific prompts following the pattern below, and provide comprehensive docstrings. 
   ```
   TaskPrompts -> YourTaskPrompts -> YourSkeletonizedTaskPrompts
   ``` 
   You must also make a {task_name}_execution-template.py.txt file under `prompts/` folder. This file contains the execution template for your task, including data loading, model initialization, and evaluation.


3. Run the agent with the standard pipeline
   
    ```bash
      python main.py \
            --output $OUTPUT_DIR \
            --dataset $DATASET_PATH \
            --gpu_id $gpu_id \
            --experiment_name $EXPERIMENT_NAME \ # For example, "cellpose_segmentation"
            --random_seed $seed \
            --history_threshold $HISTORY_THRESHOLD \ # 5
            --k $K \ # 3
            --k_word $K_WORD  # "three" 
   ```

   For MedSAM, you'll also need to download the model checkpoint [medsam_vit_b.pth](https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view?usp=drive_link) and pass in the path using `--checkpoint_path`.


## Data

To reproduce the results, download the data using the scripts and instructions found in `utils/{task_name}_data.py`.
