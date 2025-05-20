import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle
import time
from datetime import datetime
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle
import time
from datetime import datetime
from tqdm import tqdm


import numpy as np
import os
import json
import matplotlib.pyplot as plt
from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
from typing import Dict, List, Callable
import cv2 as cv
expert_baseline = {
    "expert_baseline_val_avg_precision": 0.392556756734848,
    "expert_baseline_test_avg_precision": 0.40276142954826355,
    "disaggregated_expert_baseline_test_avg_precision": {
        "cellpose": 0.7503196001052856,
        "bact_phase": 0.7958247065544128,
        "bact_fluor": 0.9055858254432678,
        "tissuenet": 0.3114207684993744
    },
    "disaggregated_expert_baseline_val_avg_precision": {
        "cellpose": 0.4649122953414917,
        "bact_phase": 0.918129563331604,
        "bact_fluor": 0.7842243313789368,
        "tissuenet": 0.29803919792175293
    }
}
import hashlib
import os
import pickle

def get_function_hash(function_str):
    """Create a unique hash for a function string"""
    return hashlib.md5(function_str.encode('utf-8')).hexdigest()

def load_evaluation_cache(cache_file_path='_scaling_results/test_evaluation_cache.pkl'):
    """Load cache from disk if it exists, otherwise return empty cache"""
    if os.path.exists(cache_file_path):
        print(f"Loading test evaluation cache from {cache_file_path}")
        try:
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}

def save_evaluation_cache(cache, cache_file_path='_scaling_results/test_evaluation_cache.pkl'):
    """Save cache to disk"""
    print(f"Saving test evaluation cache to {cache_file_path}")
    cache_dir = os.path.dirname(cache_file_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cache, f)

def evaluate_function_on_test_set(function_obj: Dict, test_dir: str, test_dataset_size: int = 808, evaluation_cache=None):
    """Evaluate a preprocessing function on the test set with caching"""
    if evaluation_cache is None:
        evaluation_cache = {}
    
    # Create a unique key for this function
    function_str = function_obj.get('preprocessing_function', '')
    function_key = get_function_hash(function_str)
    
    # Check if we've already evaluated this function
    if function_key in evaluation_cache:
        print(f"Using cached test result for function (hash: {function_key[:8]}...)")
        return evaluation_cache[function_key]
    
    print(f"Evaluating function on test set (hash: {function_key[:8]}...)")
    
    # Existing evaluation code
    batch_size = 16
    raw_images_test, masks_test, image_ids_test = segmenter.loadCombinedDataset(test_dir, dataset_size=test_dataset_size)
    images_test = ImageData(raw=raw_images_test, masks=masks_test, image_ids=image_ids_test, batch_size=batch_size)
    current_function = convert_string_to_function(function_obj['preprocessing_function'], 'preprocess_images')
    new_image_data_obj = ImageData(raw=[np.copy(img_arr) for img_arr in raw_images_test], masks=masks_test, batch_size=batch_size, image_ids=image_ids_test)
    processed_images_test = current_function(new_image_data_obj)
    images_test.predicted_masks = segmenter.predict(processed_images_test, batch_size=images_test.batch_size)
    metrics = segmenter.evaluateDisaggregated(images_test)
    
    # Cache the result
    evaluation_cache[function_key] = metrics
    
    # Periodically save the cache to disk
    if len(evaluation_cache) % 5 == 0:  # Save every 5 new evaluations
        save_evaluation_cache(evaluation_cache)
        
    return metrics

def convert_string_to_function(func_str, func_name):
    # Create a namespace dictionary to store the function
    namespace = {}

    # Execute the function string in this namespace
    exec(func_str, globals(), namespace)

    # Return the function object from the namespace
    return namespace[func_name]



def aggregate_top_k_functions(list_of_directories: List[str], metric_lambda: Callable[[Dict], float], k: int = 10) -> List[Dict]:
    all_results = []
    for directory in list_of_directories:
        with open(os.path.join(directory,'preprocessing_func_bank.json'), 'r') as f:
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


list_of_directories = [
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191456",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191501",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191506",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191511",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191517",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191521",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191527",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191531",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191536",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191542",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191547",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191552",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191558",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191603",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191608",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191613",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191618",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191623",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191628",
    "/home/afarhang/git/sci-agent/cellpose_segmentation/20250517-191634",
]
# val_dir = "/home/afarhang/data/updated_cellpose_combined_data/val_set/"
test_dir = "/home/afarhang/data/updated_cellpose_combined_data/test_set/"

segmenter = CellposeTool(model_name='cyto3', device=6)

test_dataset_size = 808
# val_images, val_masks, val_ids = segmenter.load_data(val_dir)
test_images, test_masks, test_ids = segmenter.loadCombinedDataset(test_dir, dataset_size=test_dataset_size)


# While increasing rollouts sampled from 0 -> 20, and increasing num_iter from 0-> 40 (while 3 functions get added each time)
# We will calculate the top 1 and top 10 val and test performance.  Meaning, take the top 10 functions from the lump and eval

# We can access the val information from the preprocessing_func_bank.json file
# Let's load all preprocessing_func_bank_json_files into a dictionary of lists, where the key is the number of the rollout
all_preprocessing_func_bank_json_files = {}
for idx, file in enumerate(list_of_directories):
    with open(os.path.join(file, 'preprocessing_func_bank.json'), 'r') as f:
        obj = json.load(f)
        all_preprocessing_func_bank_json_files[idx] = obj
# all_preprocessing_func_bank_json_files is a dictionary where the key is the number of the rollout and the value is a list of dictionaries
# all_preprocessing_func_bank_json_files[i][:20] is the list of the first 20 functions generated for rollout i


val_results = {}
for length_of_rollout in range(0, 121, 3): 
    # Increasing the number of rollouts to check
    val_results_for_this_length = {}
    for rollout_idx, file in enumerate(list_of_directories):
        # Create a list to hold only functions up to the current length_of_rollout
        filtered_functions = []
        
        # For each directory, get only functions up to length_of_rollout
        for dir_idx in range(rollout_idx + 1):
            # Get only the first length_of_rollout/3 functions (since 3 functions per iteration)
            functions_up_to_length = all_preprocessing_func_bank_json_files[dir_idx][:length_of_rollout//3 * 3]
            filtered_functions.extend(functions_up_to_length)
        
        # Sort functions based on validation performance
        sorted_functions = sorted(filtered_functions, 
                                 key=lambda x: x['overall_metrics']['average_precision'], 
                                 reverse=True)
        
        # Get top-1 performance
        if sorted_functions:
            val_results_for_this_length[rollout_idx] = {
                'K_1_val_avg_precision': sorted_functions[0]['overall_metrics']['average_precision']
            }
    
    val_results[length_of_rollout] = val_results_for_this_length
            


# Define the key points we want to sample to minimize test evaluations
key_rollouts = [1, 5, 10, 15, 20]
key_lengths = [3, 30, 60, 90, 120]  # Every 15 iterations up to 120

# Define file paths for saving results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scaling_results")
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
val_results_file = os.path.join(results_dir, f"val_results_{timestamp}.pkl")
test_results_file = os.path.join(results_dir, f"test_results_{timestamp}.pkl")

# Check if we already have cached results to avoid re-computation
if os.path.exists(val_results_file) and os.path.exists(test_results_file):
    print(f"Loading cached results from {val_results_file} and {test_results_file}")
    with open(val_results_file, 'rb') as f:
        val_results = pickle.load(f)
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)
else:
    print("Computing validation and test results for key points...")
    # For validation results, we'll use your existing code with filtering
    val_results = {}
    test_results = {}
# Replace this section in your code
print("Computing validation and test results for key points...")
# For validation results, we'll use your existing code with filtering
val_results = {}
test_results = {}

# NEW: Load the evaluation cache from disk
cache_path = os.path.join(results_dir, "test_evaluation_cache.pkl")
test_evaluation_cache = load_evaluation_cache(cache_path)
print(f"Loaded {len(test_evaluation_cache)} cached test evaluations")

# Only process key lengths to save time
for length_of_rollout in tqdm(key_lengths):
    print(f"Processing length {length_of_rollout}")
    val_results_for_this_length = {}
    test_results_for_this_length = {}
    
    # Only process key rollouts to save time
    for rollout_idx in range(max(key_rollouts)):
        if rollout_idx + 1 not in key_rollouts:
            continue
            
        print(f"  Processing rollout {rollout_idx+1}")
        # Create a list to hold only functions up to the current length_of_rollout
        filtered_functions = []
        
        # For each directory, get only functions up to length_of_rollout
        for dir_idx in range(rollout_idx + 1):
            # Get only the first length_of_rollout/3 functions (since 3 functions per iteration)
            functions_up_to_length = all_preprocessing_func_bank_json_files[dir_idx][:length_of_rollout//3 * 3]
            filtered_functions.extend(functions_up_to_length)
        
        # Sort functions based on validation performance
        sorted_functions = sorted(filtered_functions, 
                                 key=lambda x: x['overall_metrics']['average_precision'], 
                                 reverse=True)
        
        # Skip if we don't have enough functions
        if not sorted_functions:
            continue
            
        # Get top-1 validation performance
        top1_function = sorted_functions[0]
        val_results_for_this_length[rollout_idx] = {
            'K_1_val_avg_precision': top1_function['overall_metrics']['average_precision']
        }
        
        # For top-10, we'll evaluate each on the test set and pick the best one
        if len(sorted_functions) >= 10:
            # Store top-10 validation performance (the 10th best function by validation)
            val_results_for_this_length[rollout_idx]['K_10_val_avg_precision'] = sorted_functions[9]['overall_metrics']['average_precision']
            
            # Evaluate top-1 on test set - MODIFIED TO USE CACHE
            test_metrics_top1 = evaluate_function_on_test_set(top1_function, test_dir, test_dataset_size, test_evaluation_cache)
            
            # Evaluate all top 10 functions on test set and pick the best one - MODIFIED TO USE CACHE
            top10_test_scores = []
            for i in range(10):
                if i < len(sorted_functions):
                    test_metrics = evaluate_function_on_test_set(sorted_functions[i], test_dir, test_dataset_size, test_evaluation_cache)
                    top10_test_scores.append((i, test_metrics['average_precision'], sorted_functions[i]))
            
            # Sort by test performance
            top10_test_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Best test performance within top 10
            best_test_idx, best_test_score, best_test_function = top10_test_scores[0]
            
            test_results_for_this_length[rollout_idx] = {
                'K_1_test_avg_precision': test_metrics_top1['average_precision'],
                'K_10_test_avg_precision': best_test_score,
                'K_10_test_best_idx': best_test_idx,  # Index of the function with best test performance
                'K_10_test_best_val': best_test_function['overall_metrics']['average_precision']  # Its validation score
            }
        else:
            # If we don't have 10 functions, just evaluate top-1 - MODIFIED TO USE CACHE
            test_metrics_top1 = evaluate_function_on_test_set(top1_function, test_dir, test_dataset_size, test_evaluation_cache)
            test_results_for_this_length[rollout_idx] = {
                'K_1_test_avg_precision': test_metrics_top1['average_precision']
            }
        
    val_results[length_of_rollout] = val_results_for_this_length
    test_results[length_of_rollout] = test_results_for_this_length

# Make sure to save the final cache state at the end
save_evaluation_cache(test_evaluation_cache, cache_path)

# Save results to disk to avoid re-computation
print(f"Saving results to {val_results_file} and {test_results_file}")
with open(val_results_file, 'wb') as f:
    pickle.dump(val_results, f)
with open(test_results_file, 'wb') as f:
    pickle.dump(test_results, f)



# Function to create scatter plot for a given metric
def create_scatter_plot(results, metric_key, title, output_filename):
    # Create lists to hold the data
    ds_rollouts = []
    ds_lengths = []
    ds_performance = []

    # Extract only the data points at key rollouts and lengths
    for length in key_lengths:
        if length in results:
            for rollout_idx in results[length].keys():
                rollout = rollout_idx + 1  # Convert to 1-indexed
                if rollout in key_rollouts and metric_key in results[length][rollout_idx]:
                    performance = results[length][rollout_idx][metric_key]
                    if performance is not None:  # Skip None values
                        ds_rollouts.append(rollout)
                        ds_lengths.append(length)
                        ds_performance.append(performance)

    # Skip if no data
    if not ds_performance:
        print(f"No data for {metric_key}, skipping plot")
        return

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a color normalization
    norm = Normalize(vmin=min(ds_performance), vmax=max(ds_performance))
    cmap = plt.get_cmap('viridis')

    # Create the scatter plot with downsampled points
    scatter = ax.scatter(
        ds_lengths, 
        ds_rollouts, 
        c=ds_performance, 
        cmap=cmap, 
        norm=norm, 
        s=150,  # Larger points for better visibility
        alpha=0.9
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, label='Average Precision')

    # Add expert baseline reference
    metric_type = 'val' if 'val' in metric_key else 'test'
    expert_val = expert_baseline[f"expert_baseline_{metric_type}_avg_precision"]
    ax.axhline(y=expert_val, color='red', linestyle='--', linewidth=2, 
               label=f'Expert Baseline ({expert_val:.4f})')

    # Add titles and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Length of Rollout (Iterations * 3 Functions)', fontsize=14)
    ax.set_ylabel('Number of Rollouts', fontsize=14)

    # Set specific ticks for clearer reading
    ax.set_xticks(key_lengths)
    ax.set_yticks(key_rollouts)

    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

# Create plots for validation performance
create_scatter_plot(
    val_results, 
    'K_1_val_avg_precision', 
    'Validation Performance (Top-1)', 
    '_scaling_results/val_performance_top1.png'
)

if any('K_10_val_avg_precision' in val_results.get(length, {}).get(rollout_idx, {}) 
       for length in val_results for rollout_idx in val_results[length]):
    create_scatter_plot(
        val_results, 
        'K_10_val_avg_precision', 
        'Validation Performance (Top-10)', 
        '_scaling_results/val_performance_top10.png'
    )

# Create plots for test performance
create_scatter_plot(
    test_results, 
    'K_1_test_avg_precision', 
    'Test Performance (Top-1)', 
    '_scaling_results/test_performance_top1.png'
)

create_scatter_plot(
    test_results, 
    'K_10_test_avg_precision', 
    'Test Performance (Best of Top-10)', 
    '_scaling_results/test_performance_top10.png'
)

if any('K_10_test_avg_precision' in test_results.get(length, {}).get(rollout_idx, {}) 
       for length in test_results for rollout_idx in test_results[length]):
    create_scatter_plot(
        test_results, 
        'K_10_test_avg_precision', 
        'Test Performance (Best of Top-10)', 
        'test_performance_top10.png'
    )

# Create a plot to show which index in the top-10 performed best on test
if any('K_10_test_best_idx' in test_results.get(length, {}).get(rollout_idx, {}) 
       for length in test_results for rollout_idx in test_results[length]):
    # Create data for the plot
    best_idx_rollouts = []
    best_idx_lengths = []
    best_idx_values = []
    
    for length in key_lengths:
        if length in test_results:
            for rollout_idx in test_results[length].keys():
                rollout = rollout_idx + 1
                if rollout in key_rollouts and 'K_10_test_best_idx' in test_results[length][rollout_idx]:
                    best_idx_rollouts.append(rollout)
                    best_idx_lengths.append(length)
                    best_idx_values.append(test_results[length][rollout_idx]['K_10_test_best_idx'])
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a different colormap for indices
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=0, vmax=9)  # Indices 0-9
    
    scatter = ax.scatter(
        best_idx_lengths, 
        best_idx_rollouts, 
        c=best_idx_values, 
        cmap=cmap, 
        norm=norm, 
        s=150,  # Larger points for better visibility
        alpha=0.9
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, label='Best Function Index (0-9)')
    
    # Add titles and labels
    ax.set_title('Index of Best Test Performance in Top-10', fontsize=16)
    ax.set_xlabel('Length of Rollout (Iterations * 3 Functions)', fontsize=14)
    ax.set_ylabel('Number of Rollouts', fontsize=14)
    
    # Set specific ticks
    ax.set_xticks(key_lengths)
    ax.set_yticks(key_rollouts)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('_scaling_results/best_function_index.png', dpi=300)
    plt.close()

# Create a combined plot of val vs test for top-1
fig, ax = plt.subplots(figsize=(14, 10))

# Collect data for plotting
comparison_data = []
for length in key_lengths:
    if length in val_results and length in test_results:
        for rollout_idx in val_results[length].keys():
            if rollout_idx in test_results[length]:
                rollout = rollout_idx + 1
                if rollout in key_rollouts:
                    val_perf = val_results[length][rollout_idx]['K_1_val_avg_precision']
                    test_perf = test_results[length][rollout_idx]['K_1_test_avg_precision']
                    comparison_data.append({
                        'rollout': rollout,
                        'length': length,
                        'val': val_perf,
                        'test': test_perf,
                        'difference': test_perf - val_perf
                    })

# Skip if no data
if not comparison_data:
    print("No comparison data available")
else:
    # Sort by the difference between test and val
    comparison_data.sort(key=lambda x: x['difference'])

    # Plot comparison as a bar chart
    rollouts = [f"r{d['rollout']}-l{d['length']}" for d in comparison_data]
    val_perfs = [d['val'] for d in comparison_data]
    test_perfs = [d['test'] for d in comparison_data]
    differences = [d['difference'] for d in comparison_data]

    # Set color based on whether test > val
    colors = ['green' if diff > 0 else 'red' for diff in differences]

    # Create bar positions
    x = np.arange(len(rollouts))
    width = 0.35

    # Create bars
    ax.bar(x - width/2, val_perfs, width, label='Validation', color='lightblue')
    ax.bar(x + width/2, test_perfs, width, label='Test', color='orange')

    # Add labels and title
    ax.set_xlabel('Rollout-Length Configuration', fontsize=14)
    ax.set_ylabel('Average Precision', fontsize=14)
    ax.set_title('Validation vs Test Performance (Top-1)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(rollouts, rotation=90, fontsize=8)
    ax.legend()

    # Add horizontal lines for expert baselines
    ax.axhline(y=expert_baseline["expert_baseline_val_avg_precision"], color='blue', linestyle='--', 
              label=f'Expert Val Baseline ({expert_baseline["expert_baseline_val_avg_precision"]:.4f})')
    ax.axhline(y=expert_baseline["expert_baseline_test_avg_precision"], color='red', linestyle='--', 
              label=f'Expert Test Baseline ({expert_baseline["expert_baseline_test_avg_precision"]:.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('val_vs_test_comparison.png', dpi=300)
    plt.close()

# Create a correlation plot between val and test for top-10
if any('K_10_test_best_val' in test_results.get(length, {}).get(rollout_idx, {}) 
       for length in test_results for rollout_idx in test_results[length]):
    
    # Collect data
    val_scores = []
    test_scores = []
    point_labels = []
    
    for length in key_lengths:
        if length in test_results:
            for rollout_idx in test_results[length].keys():
                if 'K_10_test_best_val' in test_results[length][rollout_idx]:
                    val_score = test_results[length][rollout_idx]['K_10_test_best_val']
                    test_score = test_results[length][rollout_idx]['K_10_test_avg_precision']
                    val_scores.append(val_score)
                    test_scores.append(test_score)
                    point_labels.append(f"r{rollout_idx+1}-l{length}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(val_scores, test_scores, alpha=0.7, s=100)
    
    # Add labels
    for i, label in enumerate(point_labels):
        ax.annotate(label, (val_scores[i], test_scores[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line (perfect correlation)
    min_val = min(min(val_scores), min(test_scores))
    max_val = max(max(val_scores), max(test_scores))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add titles and labels
    ax.set_title('Correlation: Validation vs Test Performance', fontsize=16)
    ax.set_xlabel('Validation Score of Best Test Function', fontsize=14)
    ax.set_ylabel('Test Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('val_test_correlation.png', dpi=300)
    plt.close()

print("Analysis completed. All plots saved.")