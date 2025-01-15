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

## Notes
Autogen has an issue with sending images to Gemini, the issue ticket (with a quick fix) can be found [here](https://github.com/microsoft/autogen/issues/5033).
It turns out that `autogen` is not their official package. For `autogen-agentchat==0.2`, you can do `import autogen` and similar func is supported, see [here](https://microsoft.github.io/autogen/0.2/docs/Getting-Started#quickstart). But with the latest version 0.4.1, the APIs have been changed. 
They suggest uninstall `autogen` and use the latest version of `autogen-agentchat` and `autogen-ext`. Detailed discussion can be found in the issue ticket.

I will continue with `autoge` and the fix for now. Will clean up things later.

