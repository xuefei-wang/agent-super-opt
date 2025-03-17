# Sci-agent
Automate Scientific Data Analysis using LLM agents

## Docker Setup
```bash
# Build image from Dockerfile. You only need to run this once. 
docker build -t sciseek .

# Start the Docker container. This will output the container ID.
docker run -d -p 8888:8888 sciseek

# Open an interative terminal sesison inside your running Docker container.
docker exec -it <container_id> /bin/bash

# To run the Jupyter server, copy the link from the output into your web browser.
# The link will look like this: http://127.0.0.1:8888/tree?token=<some_token>
docker logs <container_id>

# Stop the container
docker stop <container_id>
```

## Setup

We will use virtual environment for this project. There are two requirements files - one shared and one task-dependent. 
You should add your task pacakges in the `requirements_specific.txt` to run the following commands to set up the environment.

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Merge two requirement files into one
rm -f requirements.txt
cat requirements_shared.txt requirements_specific.txt > requirements.txt

# Install packages
pip install -r requirements.txt
```

## Getting Started

All commands should be executed from the project's root directory.
You data should be located outside the repo. The data path will be provided when initializing the task prompts.
You can find the attribute `dataset_path` under `TaskPrompts`.

0. Implement your tool
   
   Create a file under `src/`, wrap your tool following the pattern below, and provide comprehensive docstrings. 
   ```
   BaseTool -> BaseTaskTool -> YourTaskTool
   ```

1. Test that your tool works as expected
   
   Basic version of tests is a manual pipeline testing that your tools works as expected. Additional unit tests are at your discretion. You can create a file under `tests/` folder and run the following command:
   
    ```bash
    python -m tests.test_X
    ```

2. Collect the docstrings
   
   The following commands will collect the doctrings from `src/` into a markdown file. (For this iteration, we are not using this, but will in the future.)
    
    ```bash
    # Default, non-recursively
    python collect_docstrings.py --input_path src --output_file artifacts/docs.md

    # Alternative, recursively (collect docstrings from all subfolders)
    python collect_docstrings.py --input_path src --output_file artifacts/docs.md --recursive
    ```

3. Intialize the function bank
   
   create an output folder and initialize an empty json file as the funtion bank:
   
    ```bash
    mkdir output
    echo "[]" > output/preprocessing_func_bank.json
    ```

4. Run the agent
   
    ```bash
    python main.py --dataset YOUR_DATA_PATH --output YOUR_OUTPUT_FOLDER
    ```
