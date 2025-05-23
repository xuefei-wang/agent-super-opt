# Figures

We provide some example code to visualize the performance of agent search, as well as visualize how the generated preprocessing function transforms the input images.

Use `fb_analysis.py` to create a plot of the performances of each generated function in the function bank and the best performing one up until that point.

Use `example_image.ipynb` to quickly visualize image transformations.

### MedSAM-Specific Files

#### `medsam_analyze_trajectories.py`
Visualizes agent trajectories for the MedSAM task. You can run this script independently or call it from `main.py` to auto-generate visualizations after agent search.

**Outputs** (saved in `figs/medsam_figs/analyze_trajectories/`):
1. **Bar Plot** – Agent vs. baseline test performance.
2. **Line Plot** – Visualizes agent validation performance over iterations in comparison to the validation baseline.
3. **Scatter Plot** – Validation vs. test performance for all `K` functions.

#### `medsam_get_open_cv_stats.py`
Visualizes the OpenCV functions common among the best vs. worst-performing functions found by the agent. Output saved as `figs/medsam_figs/supplemental/medsam_open_cv_stats.png`.

#### `medsam_supplemental_visualizations.ipynb`
Generates figures used in the publication's supplemental material. Outputs are saved in `figs/medsam_figs/supplemental/`.