# Sci-agent
Automate Scientific Data Analysis using LLM agents

All code will be run inside a Docker container.  Mount paths to data, codebase, and output; gpus enabled.  Data and output directory should not be within the codebase folder.

## Docker Setup: 
### One time per image (```Dockerfile.pytorch_gpu_cmdline``` or ```Dockerfile.pytorch_gpu_jupyter```)
```bash
# Build the image
docker build -f Dockerfile.pytorch_gpu_cmdline -t sciseek-gpu .
# or
docker build -f Dockerfile.pytorch_gpu_jupyter -t sciseek-gpu-jupyter .
```

## Running Docker container
```bash
# Start the Docker container
docker run -d --name sciseek-container-cmdline --gpus all \
  -e AUTOGEN_CACHE_DIR=/workspace/output/cache \
  -e PYTHONPATH=/workspace/repo \
  -v /path/to/your/local/repo:/workspace/repo:ro \
  -v /path/to/your/output_dir:/workspace/output:rw \
  -v /path/to/your/data:/workspace/data:ro \
  sciseek-gpu

# Enter the container
docker exec -it sciseek-container-cmdline /bin/bash

# Run main.py
cd /workspace/repo
python main.py --dataset /workspace/data/ --output /workspace/output/ --experiment_name {experiment_name}

# Exit the container 
exit

# Stop and remove container
docker stop sciseek-container-cmdline
docker rm sciseek-container-cmdline

# Make output directory writeable by anyone 
sudo chmod -R 777 /path/to/your/output_dir/
```

## Docker Setup (For ```Dockerfile```)
```bash
# Build image from Dockerfile. You only need to run this once. 
docker build -t sciseek .

# Start the Docker container. This will output the container ID.
docker run -d -p 8888:8888 sciseek

# Open an interactive terminal session inside your running Docker container.
docker exec -it <container_id> /bin/bash

# To run the Jupyter server, copy the link from the output into your web browser.
# The link will look like this: http://127.0.0.1:8888/tree?token=<some_token>
docker logs <container_id>

# Stop the container
docker stop <container_id>
```

## Setup (For fully local)

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