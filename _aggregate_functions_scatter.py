# Given a list of directories, each containing a json file (analysis_results/top_k_functions_results.json) with the results of the top k'(10) functions,
# aggregate the results.  Resort the results by a sorting function, and just keep the top k=10 results

import os
import json
from typing import List, Dict, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np

# Vanilla aggregate data
# data_subset = 'all'
# to_plot_disaggregated = True
# list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011141',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011146',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011151',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011156',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011201',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011206',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011211',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011216',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011221',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011226',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011231',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011237',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011241',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011247',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011252',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011257',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011302',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011307',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011313',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011318'
# ]

# Cellpose Only
# data_subset = 'cellpose_only'
# to_plot_disaggregated = False
# list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235356',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235402',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235407',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235412',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235417',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235422',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235427',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235432',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235437',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235442',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235448',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235453',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235459',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235504',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235510',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235515',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235520',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235525',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235530',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235536'
#                         ]

# bact fluor only
# data_subset = 'bact_fluor_only'
# to_plot_disaggregated = False
# list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235541',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235546',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235551',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235556',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235602',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235607',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235612',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235617',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235622',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235627',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235633',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235638',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235643',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235648',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235654',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235659',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235705',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235710',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235716',
#                         '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-235721']

# bact_phase only
# data_subset = 'bact_phase_only'
# to_plot_disaggregated = False
# list_of_directories = [
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082634',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082639',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082644',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082650',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082655',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082700',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082705',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082710',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082715',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082720',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082725',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082730',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082735',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082740',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082746',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082751',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082756',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082802',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082806',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-082811',
# ]

# Vanilla aggregated experiments with ablations:. No Library
# data_subset = 'all_data_no_library'
# to_plot_disaggregated = True
    
# list_of_directories = [
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161751',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161756',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161801',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161806',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161811',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161816',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161821',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161826',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161831',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161836',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161841',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161846',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161852',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161857',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161902',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161908',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161913',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161918',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161923',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250514-161928',
# ]

# Vanilla aggregated experiments with ablations: LLama
# data_subset = 'all_data_llama'
# to_plot_disaggregated=False
# list_of_directories = [
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000907',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000912',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000917',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000922',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000927',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000932',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000937',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000942',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000947',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000952',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-000957',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001003',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001008',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001013',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001018',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001024',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001029',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001034',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001039',
#     '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-001045',
# ]

# Vanilla aggregated experiments with no ablation, but num_optim_iter = 40
data_subset = 'all_data_num_optim_iter_40'
to_plot_disaggregated=False
list_of_directories = [
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191456',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191501',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191506',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191511',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191517',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191521',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191527',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191531',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191536',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191542',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191547',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191552',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191558',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191603',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191608',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191613',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191618',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191623',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191628',
    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191634',
]



# per task metric lambda
metric_lambda = lambda obj: obj['average_precision_val']
k = 10
k_prime = 10


# Read all json files in the directories
# Let's store them flat, so we have a list of dicts instead of a list of lists

def aggregate_top_k_functions(list_of_directories: List[str], metric_lambda: Callable[[Dict], float], k: int = 10, k_prime: int = 10) -> List[Dict]:
    all_results = []
    for directory in list_of_directories:
        with open(os.path.join(directory, 'analysis_results', 'top_k_functions_results.json'), 'r') as f:
            obj = json.load(f)
            for function in obj:
                function['source_directory'] = directory.split('/')[-1]
                all_results.append(function)

    # Sort the results by the sorting function
    def find_top_k(json_array: List[Dict], metric_lambda: Callable[[Dict], float], k: int) -> List[Dict]:
        '''Returns object containing the top k highest metric values from a list of JSON objects.'''
        sorted_results = sorted(json_array, key=metric_lambda, reverse=True)[:k]
        # add key "aggregate_rank" to each object
        for i, result in enumerate(sorted_results):
            result['aggregate_rank'] = i
        return sorted_results

    top_k_functions = find_top_k(all_results, metric_lambda, k=k)
    return top_k_functions


top_k_functions = aggregate_top_k_functions(list_of_directories, metric_lambda, k=k, k_prime=k_prime)
# Save json of top k functions
with open(f'_top_{k}_functions_{data_subset}.json', 'w') as f:
    json.dump(top_k_functions, f)


# Define distinct colors and markers for each source directory, up to 20 total color/marker pairs
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', '|', '_', '.', ',', '1', '2']

# Load expert baseline performances from each directory
# (Using the first one as reference for simplicity)
with open(os.path.join(list_of_directories[0], 'analysis_results/expert_baseline_performances.json'), 'r') as f:
    reference_expert_baseline_performances = json.load(f)

[print(f"val:, {obj['average_precision_val']}, test: {obj['average_precision_test']}\n {obj['preprocessing_function']} {obj['source_directory']}") for obj in top_k_functions]
# Create the figure
plt.figure(figsize=(12, 10))

# Get unique source directories to assign consistent colors/markers
unique_directories = sorted(list(set([func['source_directory'] for func in top_k_functions])))

# Create more unique combinations by varying color/marker pairs independently
def get_color_marker_combo(index, colors, markers):
    """Generate unique color/marker combinations for more directories than colors or markers"""
    color_idx = index % len(colors)
    marker_idx = (index // len(colors)) % len(markers)
    return colors[color_idx], markers[marker_idx]

# Assign color/marker combinations
dir_to_combo = {}
for i, dir in enumerate(unique_directories):
    color, marker = get_color_marker_combo(i, colors, markers)
    dir_to_combo[dir] = (color, marker)

# Track which directories we've already added to the legend
plotted_directories = set()

# Process each function
for function in top_k_functions:
    source_dir = function['source_directory']
    color, marker = dir_to_combo[source_dir]
    
    val_improvement = (function['average_precision_val'] - 
                      reference_expert_baseline_performances['expert_baseline_val_avg_precision'])
    test_improvement = (function['average_precision_test'] - 
                        reference_expert_baseline_performances['expert_baseline_test_avg_precision'])
    
    # Only add label if this is the first time we're plotting this directory
    label = source_dir if source_dir not in plotted_directories else None
    
    plt.scatter(val_improvement, test_improvement, 
               alpha=0.7, 
               color=color, 
               marker=marker, 
               s=100,  # marker size
               label=label)
    
    if label:
        plotted_directories.add(source_dir)

def create_square_scatter_plot(val_improvements: List[float], 
                              test_improvements: List[float],
                              colors: List[str],
                              markers: List[str],
                              labels: List[str] = None,
                              title: str = 'Scatter Plot',
                              output_filename: str = 'scatter_plot.png',
                              fixed_axis_limit: Optional[float] = None,
                              k : int = 10):
    """
    Create a square scatter plot with dynamic or fixed axis limits.
    
    Args:
        val_improvements: List of validation improvements
        test_improvements: List of test improvements
        colors: List of colors for each point
        markers: List of markers for each point
        labels: List of labels for legend (optional)
        title: Title for the plot
        output_filename: Filename for the output plot
        fixed_axis_limit: If provided, use this as the axis limit; otherwise calculate dynamically
    """
    plt.figure(figsize=(12, 10))
    
    # Plot points
    for i in range(len(val_improvements)):
        plt.scatter(val_improvements[i], test_improvements[i], 
                   alpha=0.7, 
                   color=colors[i], 
                   marker=markers[i], 
                   s=100,
                   label=labels[i] if labels and i < len(labels) else None)
    
    plt.xlabel('Validation Relative Improvement')
    plt.ylabel('Test Relative Improvement')
    plt.title(title)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add thicker lines at x=0 and y=0
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)
    
    # Determine axis limits
    if fixed_axis_limit is not None:
        axis_limit = fixed_axis_limit
    else:
        # Calculate dynamic axis limit based on data
        all_values = val_improvements + test_improvements
        if all_values:
            max_abs_val = max(abs(max(all_values)), abs(min(all_values)))
            # Add some padding (10%)
            axis_limit = max_abs_val * 1.1
            # Ensure minimum axis limit
            if axis_limit < 0.005:
                axis_limit = 0.005
            # Round up to nearest 0.005 for cleaner axis
            axis_limit = np.ceil(axis_limit / 0.005) * 0.005
        else:
            axis_limit = 0.02  # Default if no data
    
    plt.xlim([-axis_limit, axis_limit])
    plt.ylim([-axis_limit, axis_limit])
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add diagonal line
    plt.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', alpha=0.5, 
             label='Val = Test performance')
    
    # Add legend if labels provided
    if labels:
        n_legend_entries = len(set(labels)) + 1  # +1 for the Val=Test line
        ncol = min(3, (n_legend_entries + 4) // 5)
        plt.legend(ncol=ncol, loc='upper left', frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


# Calculate improvements for the main scatter plot
val_improvements = []
test_improvements = []
point_colors = []
point_markers = []
point_labels = []

for function in top_k_functions:
    source_dir = function['source_directory']
    color, marker = dir_to_combo[source_dir]
    
    val_improvement = (function['average_precision_val'] - 
                      reference_expert_baseline_performances['expert_baseline_val_avg_precision'])
    test_improvement = (function['average_precision_test'] - 
                       reference_expert_baseline_performances['expert_baseline_test_avg_precision'])
    
    val_improvements.append(val_improvement)
    test_improvements.append(test_improvement)
    point_colors.append(color)
    point_markers.append(marker)
    point_labels.append(source_dir if source_dir not in plotted_directories else None)
    
    if source_dir not in plotted_directories:
        plotted_directories.add(source_dir)

# Create the main scatter plot with dynamic axis limits
create_square_scatter_plot(
    val_improvements, 
    test_improvements,
    point_colors,
    point_markers,
    point_labels,
    title=f'Top {k} Functions from {len(list_of_directories)} Rollouts (showing {len(unique_directories)} unique rollouts)',
    output_filename=f'aggregate_scatter_plot_top_{k}_per_directory_colored_{data_subset}.png',
    fixed_axis_limit=None,  # Use dynamic axis limits
    k = k
)

print(f"Plot saved to aggregate_scatter_plot_top_{k}_per_directory_colored_{data_subset}.png")
print(f"Plotted {len(top_k_functions)} points from {len(unique_directories)} unique directories")


def plot_disaggregated_scatter(top_k_functions: List[Dict], reference_expert_baseline_performances: Dict, output_filename: str = f'disaggregated_scatter_plot_{data_subset}.png'):
    """
    Create a 2x2 grid of scatter plots showing performance on different datasets.
    
    Args:
        top_k_functions: List of top k functions with their metrics
        reference_expert_baseline_performances: Expert baseline performances dictionary
        output_filename: Filename for the output plot
    """
    plt.figure(figsize=(20, 20))
    datasets = ['cellpose', 'bact_phase', 'bact_fluor', 'tissuenet']
    
    # Calculate disaggregated improvements
    function_improvements = []
    for func_obj in top_k_functions:
        val_improvement = (func_obj['average_precision_val'] - 
                          reference_expert_baseline_performances['expert_baseline_val_avg_precision'])
        test_improvement = (func_obj['average_precision_test'] - 
                           reference_expert_baseline_performances['expert_baseline_test_avg_precision'])
        
        # Calculate disaggregated improvements
        val_disaggregated_improvements = {}
        test_disaggregated_improvements = {}
        
        for key in datasets:
            # Val improvements
            func_val = func_obj.get('disaggregated_average_precision_val', {}).get(key)
            baseline_val = reference_expert_baseline_performances.get('disaggregated_expert_baseline_val_avg_precision', {}).get(key)
            
            if func_val is not None and baseline_val is not None:
                val_disaggregated_improvements[key] = func_val - baseline_val
            else:
                val_disaggregated_improvements[key] = None
                
            # Test improvements
            func_test = func_obj.get('disaggregated_average_precision_test', {}).get(key)
            baseline_test = reference_expert_baseline_performances.get('disaggregated_expert_baseline_test_avg_precision', {}).get(key)
            
            if func_test is not None and baseline_test is not None:
                test_disaggregated_improvements[key] = func_test - baseline_test
            else:
                test_disaggregated_improvements[key] = None
        
        function_improvements.append({
            'val_improvement': val_improvement,
            'test_improvement': test_improvement,
            'val_disaggregated_improvements': val_disaggregated_improvements,
            'test_disaggregated_improvements': test_disaggregated_improvements
        })
    
    # Create scatter plots
    for idx, dataset_name in enumerate(datasets):
        plt.subplot(2, 2, idx + 1)
        plt.title(f'{dataset_name}: Top {len(top_k_functions)} Functions over Expert Baseline')
        
        # Extract disaggregated improvements for this specific dataset
        val_improvements_for_dataset = []
        test_improvements_for_dataset = []
        
        for func_improvement in function_improvements:
            val_imp = func_improvement['val_disaggregated_improvements'].get(dataset_name)
            test_imp = func_improvement['test_disaggregated_improvements'].get(dataset_name)
            
            if val_imp is not None and test_imp is not None:
                val_improvements_for_dataset.append(val_imp)
                test_improvements_for_dataset.append(test_imp)
        
        if val_improvements_for_dataset:
            plt.scatter(val_improvements_for_dataset, test_improvements_for_dataset, alpha=0.7)
            
            # Determine axis limits for this specific dataset
            all_values = val_improvements_for_dataset + test_improvements_for_dataset
            max_abs_val = max(abs(max(all_values)), abs(min(all_values)))
            # Add some padding (10%) to the maximum absolute value
            axis_limit = max_abs_val * 1.1
            # Ensure minimum axis limit if values are very small
            if axis_limit < 0.005:
                axis_limit = 0.005
            # Round up to nearest 0.005 for cleaner axis
            axis_limit = np.ceil(axis_limit / 0.005) * 0.005
        else:
            axis_limit = 0.02  # Default if no data
        
        # Apply formatting
        plt.xlabel('Validation Relative Improvement')
        plt.ylabel('Test Relative Improvement')
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add thicker lines at x=0 and y=0
        plt.axhline(0, color='black', linewidth=1.5)
        plt.axvline(0, color='black', linewidth=1.5)
        
        # Set axis limits to make it square
        plt.xlim([-axis_limit, axis_limit])
        plt.ylim([-axis_limit, axis_limit])
        
        # Make the axes perfectly square
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Add a diagonal line to show where val=test
        plt.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', alpha=0.5, 
                label='Val = Test performance')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_disaggregated_test_performance(top_k_functions: List[Dict], 
                                       reference_expert_baseline_performances: Dict,
                                       include_no_preprocessing: bool = True,
                                       output_filename: str = 'disaggregated_test_performance.png'):
    """
    Create bar charts showing test performance on different datasets.
    
    Args:
        top_k_functions: List of top k functions with their metrics
        reference_expert_baseline_performances: Expert baseline performances dictionary
        include_no_preprocessing: Whether to include the no preprocessing baseline
        output_filename: Filename for the output plot
    """
    plt.figure(figsize=(20, 20))
    datasets = ['cellpose', 'bact_phase', 'bact_fluor', 'tissuenet']
    
    # Find best performing function on test
    best_function = max(top_k_functions, key=lambda x: x['average_precision_test'])
    
    for idx, dataset_name in enumerate(datasets):
        plt.subplot(2, 2, idx + 1)
        
        # Get values and format them
        expert_val = reference_expert_baseline_performances.get('disaggregated_expert_baseline_test_avg_precision', {}).get(dataset_name)
        agent_val = best_function.get('disaggregated_average_precision_test', {}).get(dataset_name)
        
        # Format labels
        expert_label = f"Expert Baseline: {expert_val:.4f}" if expert_val is not None else "Expert Baseline: N/A"
        agent_label = f"Agent Designed: {agent_val:.4f}" if agent_val is not None else "Agent Designed: N/A"
        
        # Use 0 for None values for visualization
        expert_val = expert_val if expert_val is not None else 0
        agent_val = agent_val if agent_val is not None else 0
        
        # Plot bars
        bars = []
        labels = []
        colors = []
        
        bars.extend(['expert_baseline', 'agent_designed'])
        labels.extend([expert_label, agent_label])
        colors.extend(['b', 'r'])
        
        if include_no_preprocessing:
            # Note: In the aggregated data, we don't have the no preprocessing baseline
            # So we'll add a placeholder or you can modify to load this data separately
            baseline_label = "No Preprocessing: N/A"
            bars.append('no_preprocessing')
            labels.append(baseline_label)
            colors.append('k')
        
        # Create the bars
        bar_values = [expert_val, agent_val]
        if include_no_preprocessing:
            bar_values.append(0)  # Placeholder - you would need to load actual data
        
        plt.bar(bars[:len(bar_values)], bar_values, color=colors[:len(bar_values)])
        
        # Add labels on bars
        for i, (bar, label) in enumerate(zip(bars[:len(bar_values)], labels[:len(bar_values)])):
            plt.text(i, 0.001, label, ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.xlabel('Method')
        plt.ylabel('Average Precision')
        plt.title(f'{dataset_name} - Test Set Performance')
        plt.xticks([])  # Remove x-axis labels since we're using text labels
        
        # Set y-axis limits based on values
        if any(v > 0 for v in bar_values):
            plt.ylim(0, max(bar_values) * 1.2)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()



if to_plot_disaggregated:
    # Now let's call these functions to create the plots
    plot_disaggregated_scatter(top_k_functions, reference_expert_baseline_performances, 
                            f'aggregate_disaggregated_scatter_plot_{data_subset}.png')

    plot_disaggregated_test_performance(top_k_functions, reference_expert_baseline_performances,
                                    include_no_preprocessing=False,  # Set to False since we don't have this data
                                    output_filename=f'aggregate_disaggregated_test_performance_{data_subset}.png')

print("Created disaggregated scatter plot: aggregate_disaggregated_scatter_plot.png")
print("Created disaggregated test performance plot: aggregate_disaggregated_test_performance.png")