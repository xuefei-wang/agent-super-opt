import matplotlib.pyplot as plt
import numpy as np
import os
import json
# first 8
list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032611',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032645',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032721',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032756',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032831',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032907',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-032943',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033016'
                        ]

## second 8
# list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033054',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033129',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033204',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033239',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033318',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033352',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250511-033428',
#                         '/home/afarhang/git/sci-agent/cellpose_segme ntation/20250511-033504']


# Our choice of k, default 5. Select the top k functions from each directory
k = 5
# Load expert baseline performances from each directory
# (Using the first one as reference for simplicity)
with open(os.path.join(list_of_directories[0], 'analysis_results/expert_baseline_performances.json'), 'r') as f:
    reference_expert_baseline_performances = json.load(f)

# Initialize lists to hold all points we want to plot
all_val_relative_improvements = []
all_test_relative_improvements = []

# Process each directory
for directory in list_of_directories:
    # Load top_k_functions_results.json from the directory
    with open(os.path.join(directory, 'analysis_results/top_k_functions_results.json'), 'r') as f:
        directory_functions = json.load(f)
    
    # Calculate improvement for each function in this directory
    directory_improvements = []
    for func_obj in directory_functions:
        val_improvement = (func_obj['average_precision_val'] - 
                          reference_expert_baseline_performances['expert_baseline_val_avg_precision'])
        test_improvement = (func_obj['average_precision_test'] - 
                           reference_expert_baseline_performances['expert_baseline_test_avg_precision'])
        
        # Store the improvements along with function info for sorting
        directory_improvements.append({
            'val_improvement': val_improvement,
            'test_improvement': test_improvement,
            'function_name': func_obj.get('function_name', 'unknown')
        })
    
    # Sort by validation improvement (descending)
    sorted_improvements = sorted(directory_improvements, 
                               key=lambda x: x['val_improvement'], 
                               reverse=True)
    
    # Take the top k (or all if less than k) from this directory
    top_k = sorted_improvements[:min(k, len(sorted_improvements))]
    
    # Add to our plotting lists
    for func in top_k:
        all_val_relative_improvements.append(func['val_improvement'])
        all_test_relative_improvements.append(func['test_improvement'])
        print(f"Directory: {os.path.basename(directory)}, Function: {func['function_name']}, "
              f"Val improvement: {func['val_improvement']:.4f}, Test improvement: {func['test_improvement']:.4f}")

# Now we have the top k from each directory in our plotting lists
print(f"Total points to plot: {len(all_val_relative_improvements)}")

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(all_val_relative_improvements, all_test_relative_improvements, alpha=0.7)
plt.xlabel('Validation Relative Improvement')
plt.ylabel('Test Relative Improvement')
plt.title(f'Top {k} Functions per Directory for {len(list_of_directories)} Directories over Expert Baseline')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Add thicker lines at x=0 and y=0
plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

# Determine axis limits to make it a centered square
axis_limit = 0.02  # Fixed limit as requested

plt.xlim([-axis_limit, axis_limit])
plt.ylim([-axis_limit, axis_limit])
plt.gca().set_aspect('equal', adjustable='box')

# Add a diagonal line to show where val=test
plt.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', alpha=0.5, 
         label='Val = Test performance')
plt.legend()

plt.savefig(f'scatter_plot_top_{k}_per_directory.png')
print(f"Plot saved to scatter_plot_top_{k}_per_directory.png")