import ast
import json

import cv2 as cv
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

all_folders = os.listdir('../output/dermoscopy-main/medSAM_segmentation')
good_folders = [f for f in all_folders if '-' in f]
list_of_directories = [os.path.join('../output/dermoscopy-main/medSAM_segmentation', f) for f in good_folders]

class CV2FunctionExtractor(ast.NodeTransformer):
    def __init__(self):
        self.func_calls = []

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"cv", "cv2"}
        ):
            func_name = node.func.attr
            self.func_calls.append(func_name)
        return self.generic_visit(node)

def find_opencv_functions(func_str):
    tree = ast.parse(func_str)
    transformer = CV2FunctionExtractor()
    modified_tree = transformer.visit(tree)
    return transformer.func_calls

def parse_function_bank(directory):
    ''' Returns frequency of all functions in a rollout, top 10 and bottom 10 function calls and val score, and top test score'''
    with open(os.path.join(directory, 'preprocessing_func_bank.json'), 'r') as f:
        curr_iter_list = json.load(f)
        all_functions_in_rollout = []
        for obj in curr_iter_list:
            function_calls = find_opencv_functions(obj['preprocessing_function'])
            
            all_functions_in_rollout.extend(function_calls)
        
        # Top 10 on val
        top_10 = sorted(curr_iter_list, key=lambda x: x['overall_metrics']['nsd_metric'] + x['overall_metrics']['dsc_metric'], reverse=True)[:10]
        
        top_10_analysis = []
        for top in top_10:
            top_functions = find_opencv_functions(top['preprocessing_function'])
            top_score = top['overall_metrics']['nsd_metric'] + top['overall_metrics']['dsc_metric']
            top_10_analysis.append({
                'function_calls': top_functions,
                'combined_score': top_score
            })
        
        
        # Bottom 10 on val
        bottom_10 = sorted(curr_iter_list, key=lambda x: x['overall_metrics']['nsd_metric'] + x['overall_metrics']['dsc_metric'])[:10]
        
        bottom_10_analysis = []
        for bottom in bottom_10:
            bottom_functions = find_opencv_functions(bottom['preprocessing_function'])
            bottom_score = bottom['overall_metrics']['nsd_metric'] + bottom['overall_metrics']['dsc_metric']
            bottom_10_analysis.append({
                'function_calls': bottom_functions,
                'average_precision': bottom_score
            })
    
    # Top on test
    with open(os.path.join(directory, 'analysis_results/top_k_functions_results.json'), 'r') as f:
        top_k = json.load(f)
        top_test = 0 
        for i in range(len(top_k)):
            top_test = max(top_k[i]['combined_test'], top_test)
    
    return Counter(all_functions_in_rollout), top_10_analysis, bottom_10_analysis, top_test
        
all_parsed_data = []
for i in list_of_directories:
    all_parsed_data.append(parse_function_bank(i))

print('done')


total_counter_top_10 = Counter()
total_counter_bottom_10 = Counter()
# Parse distribution of top 10 functions
for idx, parsed_data in enumerate(all_parsed_data):
    top_10_analysis = parsed_data[1]
    counters = [Counter(top_10_analysis[i]['function_calls']) for i in range(10)]

    # Use Counter addition to combine them
    for counter in counters:
        total_counter_top_10 += counter

    bottom_10_analysis = parsed_data[2]
    counters = [Counter(bottom_10_analysis[i]['function_calls']) for i in range(10)]

    for counter in counters:
        total_counter_bottom_10 += counter


print('done')

# Assuming you already have these Counter objects from your code
# total_counter_top_10 and total_counter_bottom_10
def plot_function_frequencies(top_counter, bottom_counter, use_percentage=True):
    # Get all unique function names
    all_functions = set(list(top_counter.keys()) + list(bottom_counter.keys()))
    
    # Calculate total counts for normalization
    top_total = sum(top_counter.values())
    bottom_total = sum(bottom_counter.values())
    
    # Create lists for plotting
    functions = list(all_functions)
    
    # Convert counts to frequencies/percentages
    if top_total > 0 and bottom_total > 0:
        top_freqs = [top_counter.get(func, 0) / top_total for func in functions]
        bottom_freqs = [bottom_counter.get(func, 0) / bottom_total for func in functions]
        
        # Convert to percentages if requested
        if use_percentage:
            top_freqs = [freq * 100 for freq in top_freqs]
            bottom_freqs = [freq * 100 for freq in bottom_freqs]
    else:
        top_freqs = [0] * len(functions)
        bottom_freqs = [0] * len(functions)
    
    # Sort by total frequency (descending)
    sorted_indices = np.argsort([t + b for t, b in zip(top_freqs, bottom_freqs)])[::-1]
    functions = [functions[i] for i in sorted_indices]
    top_freqs = [top_freqs[i] for i in sorted_indices]
    bottom_freqs = [bottom_freqs[i] for i in sorted_indices]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on X axis
    r1 = np.arange(len(functions))
    r2 = [x + bar_width for x in r1]
    
    # Make the plot
    ax.bar(r1, top_freqs, width=bar_width, label='Top 10', color='cornflowerblue', alpha=0.8)
    ax.bar(r2, bottom_freqs, width=bar_width, label='Bottom 10', color='salmon', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('OpenCV Functions', fontweight='bold', fontsize=12)
    
    # Set y-axis label based on whether we're showing percentages or proportions
    if use_percentage:
        ax.set_ylabel('Frequency (%)', fontweight='bold', fontsize=12)
    else:
        ax.set_ylabel('Frequency (proportion)', fontweight='bold', fontsize=12)
    
    ax.set_title('Frequency of OpenCV Functions in Top 10 vs Bottom 10', fontweight='bold', fontsize=14)
    
    # Add xticks on the middle of the group bars
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(functions, rotation=45, ha='right')
    
    # Create legend & show graphic
    ax.legend()
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Show plot
    fig_output_path = '_frequencies/opencv_function_frequencies.png'
    os.makedirs(os.path.dirname(fig_output_path), exist_ok=True)
    plt.savefig(fig_output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# For percentages (default)
plot_function_frequencies(total_counter_top_10, total_counter_bottom_10, use_percentage=True)
