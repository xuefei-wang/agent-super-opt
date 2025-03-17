# Sci-agent
Automate Scientific Data Analysis using LLM agents

## Setup

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

First, collect all docstrings: 
```bash
# Default, non-recursively
python collect_docstrings.py --input_path src --output_file artifacts/docs.md

# Alternative, recursively (collect docstrings from all subfolders)
python collect_docstrings.py --input_path src --output_file artifacts/docs.md --recursive
```

Next, create an output folder and initialize an empty json file as the funtion bank:
```bash
mkdir output
echo "[]" > output/preprocessing_func_bank.json
```

Then, to run the agent workflow:
```bash
python main_agent.py
```

If you want to run tests:
```bash
python -m tests.test_X
```

If you want to launch experiments using shell script: 
```bash
bash setup_experiment.sh
```
