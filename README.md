# Sci-agent

Automate Scientific Data Analysis using LLM agents


First, collect all docstrings: 
```bash
# non-recursively
python collect_docstrings.py --input_path src --output_file intermediate-data/docs.md

# recursively
python collect_docstrings.py --input_path src --output_file intermediate-data/docs.md --recursive
```


To run it manually:
```bash
python main_manual.py --zarr_path /data/user-data/xwang3/celltype-data-public-greenbaum-with-nuclei-channel/Tissue-Reproductive-Greenbaum_Uterus_MIBI.zarr --output output --gpu 3

```


To run it using an agent:
```bash
python main.py
```