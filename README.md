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

Set up your LLM API key as an environment variable. By default, we are using OpenAI models and therefore requires a `OPENAI_API_KEY`. You can change the backbone LLMs by updating the `llm_config` for all agents in main.py.

## Getting Started

All commands should be executed from the project's root directory.
You data should be located outside the repo. The data path will be provided in `--dataset`.

0. Implement your tool
   
   Create a file under `src/`, wrap your tool into a class and provide comprehensive docstrings. Your tool should have `__init__`, `evaluate()`, `predict()` methods.

1. Test that your tool works as expected
   
   Basic version of tests is a manual pipeline testing that your tools works as expected. Additional unit tests are at your discretion. You can create a file under `tests/` folder and run the following command:
   
    ```bash
    python -m tests.test_X
    ```

2. Create your task specific prompts class with skeletonization

   Create a task specific prompt file `{task_name}_prompts.py` under `prompts/` folder, the prompt class should inherit from `TaskPrompts` class.

   Make a `{task_name}_execution-template.py.txt` file under `prompts/` folder. This file contains the execution template for your task, including data loading, model initialization, and evaluation. Refer to `base_execution-template.py.txt` as an example. Your expert baseline function should be provided in `prompts/{task_name}_expert.py.txt`. 

   The new prompt class, along with a `sampling_function` that defines your metric for optimization should be provided in `main.py`.


3. Run the agent with the standard pipeline
   
    ```bash
      python main.py \
            --dataset $DATASET_PATH \
            --gpu_id $gpu_id \
            --experiment_name $EXPERIMENT_NAME \ # For example, "cellpose_segmentation"
            --random_seed $seed \
            --history_threshold $HISTORY_THRESHOLD \ # For example, 5
            --k $K \ # For example, 3
            --k_word $K_WORD  # For example, "three"
   ```

   For MedSAM, you'll also need to download the model checkpoint [medsam_vit_b.pth](https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view?usp=drive_link) and pass in the path using `--checkpoint_path`.

   More arguments can be found in `main.py`. Specifically, 
   - `history_threshold` is a parameter for when to start incorporating function bank hisotry into the prompt.
   - `k` and `k_word` is the number of samples to be generate per iteration.


## Data

To reproduce the results, download the data using the scripts and instructions found in `utils/{task_name}_data.py`.
