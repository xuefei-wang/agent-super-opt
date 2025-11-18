# Simple Agents Outperform Experts in Biomedical Imaging Workflow Optimization

## 💡 Overview

Adapting production-level computer vision tools to bespoke scientific datasets is a critical ``last mile'' bottleneck. We consider using AI agents to automate this manual coding process, and focus on the open question of optimal agent design. We introduce a systematic evaluation framework for agentic code optimization and use it to study three production-level biomedical imaging pipelines. We demonstrate that a simple agent framework consistently generates adaptation code that outperforms human-expert solutions. Our analysis reveals that common, complex agent architectures are not universally beneficial, leading to a practical roadmap for agent design. This repository contains the code for 1) Evaluate and compare different agent designs, 2) Allow scientists to add their own tasks and achieve tool adaptation.



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
      * **Polaris (Spot Detection)**: Set your `DEEPCELL_ACCESS_TOKEN` environment variable. Instructions are available [here](https://deepcell.readthedocs.io/en/master/API-key.html).
      * **Cellpose**: Need to comment out `fill_holes_and_remove_small_masks` step in the `dynamics.resize_and_compute_masks`

3.  **Run the Experiments**

    Execute the `main.py` script with the relevant arguments:

    ```bash
    python main.py \
          --dataset $DATASET_PATH \
          --gpu_id $GPU_ID \
          --experiment_name $EXPERIMENT_NAME \ # E.g., "cellpose_segmentation", "spot_detection", "medSAM_segmentation"
          --random_seed $SEED \
          --num_optim_iter $NUM_OPTIM_ITER \ # E.g., 20. How many iterations to run.
          --history_threshold $HISTORY_THRESHOLD  # E.g., 5. When to start incorporating function bank history into the prompt.
    ```

4. **Analyze the trajectories**
      
      Analyze trajectories using `figs/{task_name}_analyze_trajectories.py`. It will create a `analysis_results` folder under each timestamped result folder, evaluate the functions on test sets and generate plots. This might a while, feel free to parallize the runs if needed.


### For Scientists: Applying to Your Own Data 🧪

This guide explains how to adapt this agentic framework to optimize a workflow for your specific scientific tool and dataset.

To integrate your custom workflow, you'll need to implement the following components:

1.  **Tool Wrapper**: Create `src/{task_name}.py`. This file acts as a Python wrapper for your scientific tool, exposing `__init__`, `evaluate()`, and `predict()` methods. This needs to be a bare backbone of tool calling without expert preprocessing and postprocessing steps.

2.  **Prompts**: Define your task-specific prompts in `prompts/{task_name}_prompts.py`. This file should inherit from `TaskPrompts` and provide detailed instructions about your data, task objectives, and evaluation metrics to the AI agent.

3.  **Execution Template**: Set up an execution template in `prompts/{task_name}_execution-template.py.txt`. This template outlines the overall workflow where your tool will be used, and the agent-generated preprocessing and postprocessing functions will be "plugged in."

4.  **Expert Baseline** (Postprocessing step only): Provide an expert-written baseline implementation in `prompts/{task_name}_expert_postprocessing.py.txt`. For highly specific postprocessing steps, provide a skeleton in `prompts/{task_name}_expert_postprocessing_skeleton.py.txt` to provide minimal guidance for LLM agents.

5.  **Register Task in Main**: Add your task to the if/else branch in `main.py` (around line 404-420). You'll need to:
    - Import your prompt class from `prompts/{task_name}_prompts.py`
    - Define the `sampling_function` that extracts the optimization metric from the results saved to the function bank (e.g., `lambda x: x['overall_metrics']['metric_name']`)
    - Specify the `kwargs_for_prompt_class` with parameters your prompt class needs (e.g., `gpu_id`, `seed`, `dataset_path`, `function_bank_path`, `k`, `k_word`). These are passed to your prompt class constructor and can be used to fill `{placeholder}` values in your execution template or for generating agent prompts.
    - Add your task name to the `choices` list in the argument parser (around line 622)

6.  **AutoML Support** (Optional): If you want to enable hyperparameter optimization with the `--hyper_optimize` flag, add an evaluation branch for your task in `prompts/automl_execution_template.py.txt` (around line 166-237). This branch should:
    - Import your tool wrapper
    - Load your dataset
    - Create an `ImageData` object
    - Call the preprocessing function, model prediction, and postprocessing function
    - Return metrics in the format `{'overall_metrics': metrics}`

Once you've implemented these components, you can run the optimization:

```bash
python main.py \
      --dataset $DATASET_PATH \
      --gpu_id $GPU_ID \
      --experiment_name $YOUR_CUSTOM_TASK_NAME \
      --random_seed $SEED \
      --num_optim_iter $NUM_OPTIM_ITER \ # How many iterations to run.
      --history_threshold $HISTORY_THRESHOLD \ # When to start incorporating function bank history into the prompt.
```

## ⚙️ Understanding `main.py` Arguments

The `main.py` script is the entry point for running the framework and offers a variety of command-line arguments to customize the optimization process.

### Basic Arguments

  * `--dataset (-d)`: Path to your dataset.
  * `--experiment_name`: Name of the experiment. For reproducibility, choose from `"spot_detection"`, `"cellpose_segmentation"`, or `"medSAM_segmentation"`. For custom workflows, use the name you defined for your task.
  * `--checkpoint_path`: Path to a model checkpoint file. Currently used only for MedSAM segmentation.
  * `--gpu_id`: The ID of the GPU to use (default: `0`).
  * `--random_seed`: The random seed for reproducibility (default: `42`).
  * `--num_optim_iter`: The number of iterations to run in total (default: `20`).

### Function Bank Display Arguments

  * `--n_top`: The number of top-performing functions to display in the function bank (default: `3`).
  * `--n_worst`: The number of worst-performing functions to display in the function bank (default: `3`).
  * `--n_last`: The number of last functions to show in the function bank (default: `0`).
  * `--history_threshold`: The number of iterations to wait before incorporating the accumulated function bank history into the agent's prompt (default: `0`).

### Hyperparameter Optimization Arguments

  * `--hyper_optimize`: Enable hyperparameter optimization. When set, runs automated hyperparameter search using Optuna after function generation.
  * `--n_hyper_optimize`: Number of top-performing functions to optimize hyperparameters for (default: `3`).
  * `--n_hyper_optimize_trials`: Number of Optuna trials to run for each function during hyperparameter optimization (default: `24`).
  * `--hyper_optimize_interval`: Run hyperparameter optimization every N iterations during the optimization process (default: `5`).

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
