# Sci-agent

Automate Scientific Data Analysis using LLM agents

## Setup 

```bash
# install packages
pip install -r requirements.txt

# install sam2
cd src && \
git clone git@github.com:facebookresearch/sam2.git && \
cd sam2 && \
pip install -e . && \
cd checkpoints && \
./download_ckpts.sh && \
cd ../../..
```

## Getting Started

First, collect all docstrings: 
```bash
# non-recursively, default
python collect_docstrings.py --input_path src --output_file artifacts/docs.md

# recursively (collect docstrings from all subfolders)
python collect_docstrings.py --input_path src --output_file artifacts/docs.md --recursive
```

To run it manually:
```bash
python main_manual.py --data_path /data/user-data/xwang3/DynamicNuclearNet/DynamicNuclearNet-segmentation-v1_0/test.npz --output output --gpu 7 --segmenter mesmer
## Additional flags:
# --no-viz: Disable static visualization saving
# --interactive: Enable interactive napari visualization
```


To run it using an agent:
```bash
python main_agent.py
```

