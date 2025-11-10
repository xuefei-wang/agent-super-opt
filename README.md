# Agentic Superoptimization of Scientific Analysis Workflows

## 💡 Overview

This repository contains the code for our paper, "Agentic Superoptimization of Scientific Analysis Workflows" ([TODO: Add link when available]). This work introduces a novel paradigm where AI agents autonomously write and optimize data analysis code for complex scientific tasks. Our method generates solutions that can surpass manually-tuned, expert-written code, significantly accelerating the pace of scientific discovery.


## 🚀 Getting Started

To get started, follow these steps. We'll first cover the environment setup, which is common for all users, then dive into specific usage guides.

We strongly recommend using a virtual environment to manage dependencies for this project.

1.  **Create and Activate Virtual Environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Prepare Requirements Files**: This project uses a shared `requirements_shared.txt` and a task-specific `requirements_specific_{task_name}.txt`. You'll need to specify your task when setting up.

    ```bash
    # Replace {task_name} with your desired task (e.g., cellpose_segmentation, spot_detection, medSAM_segmentation)
    rm -f requirements.txt
    cat requirements_shared.txt requirements_specific_{task_name}.txt > requirements.txt
    ```

3.  **Install Python Packages**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up LLM API Key**: This framework uses OpenAI models by default, so you'll need to set your `OPENAI_API_KEY` as an environment variable. You can do so by creating a `.env` file. If you prefer other LLMs, you can update the `llm_config` for agents in `main.py`.


## 📖 Usage Guides

This repository offers two main modes of operation: 

* For CS/AI researchers: reproducing our paper's results

* For scientists: applying the framework to your own custom data analysis workflows

Please follow the one that best matches your goal:

### For CS/AI Researchers: Reproducing Paper Results 💻

This guide provides instructions for replicating the experimental results presented in our paper, covering Polaris, Cellpose, and MedSAM workflows.

1.  **Data Preparation**

    Download the necessary datasets using the provided scripts and instructions for each task. You'll find these in `utils/{task_name}_data.py`. The path to your downloaded dataset will be passed using the `--dataset` argument during execution.

2.  **Additional Setup (Task-Specific)**

      * **MedSAM**: Download the model checkpoint medsam\_vit\_b.pth and specify its path using `--checkpoint_path`. Follow the [instructions](https://github.com/bowang-lab/MedSAM?tab=readme-ov-file#installation) and install the MedSAM package.
      * **Polaris**: Set your `DEEPCELL_ACCESS_TOKEN` environment variable. Instructions are available [here](https://deepcell.readthedocs.io/en/master/API-key.html).
      * **Cellpose**: Need to comment out `fill_holes_and_remove_small_masks` step in the `dynamics.resize_and_compute_masks`

3.  **Run the Experiments**

    Execute the `main.py` script with the relevant arguments:

    ```bash
    python main.py \
          --dataset $DATASET_PATH \
          --gpu_id $GPU_ID \
          --experiment_name $EXPERIMENT_NAME \ # E.g., "cellpose_segmentation", "spot_detection", "medSAM_segmentation"
          --random_seed $SEED \
          --history_threshold $HISTORY_THRESHOLD \ # E.g., 5. When to start incorporating function bank history into the prompt.
          --k $K \ # E.g., 3. Number of samples (functions) to generate per iteration.
          --k_word $K_WORD # E.g., "three". The word representation of 'k'.
    ```

4. **Analyze the trajectories**

      Run the following two commands to: 
      
      First, analyze trajectories - It will create a `analysis_results` folder under each timestamped result folder, evaluate the functions on test sets and generate plots. This might a while, feel free to parallize the runs if needed.

      Second, aggregates all results into the global function json file and generate a learning curve plot.

      
      - For Cellpose: 
            
      ```python
      python figs/cellpose_analyze_trajectories.py \
            --data_path=$DATA_FOLDER
      ```

      - For Polaris:

      ```python
      python figs/spot_detection_analyze_trajectories.py \
            --checkpoint_path=$CHECKPOINT_FILE \
            --val_data_path=$VAL_DATA_FILE \
            --test_data_path=$TEST_DATA_FILE \
            --gpu_id=$GPU_ID
      ```

      - For MedSAM:
            
      ```python
      python figs/spot_detection_analyze_trajectories.py \
            --data_path=$DATA_FOLDER \ # this should be the folder where val/ and test/ are stored
            --gpu_id $GPU_ID
      ```


### For Scientists: Applying to Your Own Data 🧪

This guide explains how to adapt this agentic framework to optimize a data preprocessing workflow for your specific scientific tool and dataset.

To integrate your custom workflow, you'll need to implement the following components:

1.  **Tool Wrapper**: Create `src/{task_name}.py`. This file acts as a Python wrapper for your scientific tool, exposing `__init__`, `evaluate()`, and `predict()` methods.

      * **Example**: See `src/cellpose_segmentation.py` for a reference implementation.

2.  **Prompts**: Define your task-specific prompts in `prompts/{task_name}_prompts.py`. This file should inherit from `TaskPrompts` and provide detailed instructions about your data, task objectives, and evaluation metrics to the AI agent.

      * **Example**: Refer to `prompts/cellpose_segmentation_prompts.py`.

3.  **Execution Template**: Set up an execution template in `prompts/{task_name}_execution-template.py.txt`. This template outlines the overall workflow where your tool will be used, and the agent-generated preprocessing functions will be "plugged in."

      * **Example**: Check `prompts/cellpose_segmentation_execution-template.py.txt`.

4.  **Expert Baseline**: Provide an expert-written baseline implementation in `prompts/{task_name}_expert_postprocessing.py.txt`. For highly specific postprocessing steps, provide a skeleton in `prompts/{task_name}_expert_postprocessing_skeleton.py.txt` to provide minimal guidance for LLM agents.

      * **Example**: See `prompts/medsam_segmentation_expert_postprocessing_skeleton.py.txt`.

Once you've implemented these components, you can run the optimization:

```bash
python main.py \
      --dataset $DATASET_PATH \
      --gpu_id $GPU_ID \
      --experiment_name $YOUR_CUSTOM_TASK_NAME \
      --random_seed $SEED \
      --history_threshold $HISTORY_THRESHOLD \ # When to start incorporating function bank history into the prompt.
      --k $K \ # Number of samples (functions) to generate per iteration (default: 3).
      --k_word $K_WORD # The word representation of 'k' (default: "three").
```

## ⚙️ Understanding `main.py` Arguments

The `main.py` script is the entry point for running the framework and offers a variety of command-line arguments to customize the optimization process.

  * `--dataset (-d)`: Path to your dataset.
  * `--output (-o)`: Path to the output folder where all results will be saved.
  * `--experiment_name`: Name of the experiment. For reproducibility, choose from `"spot_detection"`, `"cellpose_segmentation"`, or `"medSAM_segmentation"`. For custom workflows, use the name you defined for your task.
  * `--checkpoint_path`: Path to a model checkpoint file. Currently used only for MedSAM segmentation.
  * `--gpu_id`: The ID of the GPU to use (default: `0`).
  * `--random_seed`: The random seed for reproducibility (default: `42`).
  * `--n_top`: The number of top-performing functions to display in the function bank (default: `3`).
  * `--n_worst`: The number of worst-performing functions to display in the function bank (default: `3`).
  * `--n_last`: The number of last functions to show in the function bank (default: `0`).
  * `--history_threshold`: The number of iterations to wait before incorporating the accumulated function bank history into the agent's prompt (default: `5`).
  * `--k`: The preprocessing function group size, representing the number of new functions to generate per iteration (default: `3`).
  * `--k_word`: The English word representation of `k` (e.g., `"three"`) (default: `"three"`). This is used for prompt phrasing.

## 📦 Output Structure

Upon successful execution, you will find results saved into your designated output directory. Experiments folders are timestamped. Inside the folder, you can find the preprocessing functions stored in a JSON file:

`{task_name}/preprocessing_func_bank.json`

This file contains the "function bank" of all functions explored and their associated performance metrics throughout the optimization process.

## 📄 License

This project is licensed under the Apache License 2.0.


## Citation 

```
[Insert citation once available]
```