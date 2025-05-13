import json
import os
import json
from typing import List, Dict, Callable
import numpy as np
import cv2 as cv
import pandas as pd
import argparse
import sys

_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.medsam_segmentation import MedSAMTool
from src.data_io import ImageData
import io
import contextlib
import csv
import matplotlib.pyplot as plt

def get_baseline(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='replace') as file:
        baseline_data = file.readlines()
        baseline_dsc = [float(line.split(": ")[1]) for line in baseline_data if "Average DSC metric" in line][0]
        baseline_nsd = [float(line.split(": ")[1]) for line in baseline_data if "Average NSD metric" in line][0]
    return baseline_dsc, baseline_nsd, baseline_dsc + baseline_nsd

def find_top_k(json_array: List[Dict], metric_lambda: Callable[[Dict], float], k=5) -> Dict:
    '''Returns object containing the top k highest metric values from a list of JSON objects.'''
    return sorted(json_array, key=metric_lambda, reverse=True)[:k]

def convert_string_to_function(func_str, func_name):
    # Create a namespace dictionary to store the function
    namespace = {}

    # Execute the function string in this namespace
    exec(func_str, globals(), namespace)

    # Return the function object from the namespace
    return namespace[func_name]

def find_all_metrics(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> List[float]:
    '''Returns a list of metric values from a list of JSON objects.'''
    return [metric_lambda(obj) for obj in json_array]

def find_highest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns object with the highest metric value from a list of JSON objects.'''
    return max(json_array, key=metric_lambda)

def find_rolling_highest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns a list of metric values, each index being the highest value up until that point'''
    rolling_highest = []
    current_highest = float('-inf')
    for obj in json_array:
        metric_value = metric_lambda(obj)
        if metric_value > current_highest:
            current_highest = metric_value
        rolling_highest.append(current_highest)
    return rolling_highest

def dump_functions_to_txt(json_array: List[Dict], metric_lambda: Callable[[Dict], float], output_path: str):
    '''Print preprocessing functions and their metric values to a text file for readability'''
    with open(output_path, 'w') as file:
        for obj in json_array:
            metric_value = metric_lambda(obj)
            file.write('\n')
            file.write(f'Value: {metric_value}\n')
            file.write(obj['preprocessing_function'])
            file.write('\n')

def get_new_json(json_path):
    with open(json_path, 'r') as file:
        json_array = json.load(file)

    # Handle json ambiguities
    new_json = []
    for i in range(len(json_array)):
        data_for_json = {'preprocessing_function': json_array[i]['preprocessing_function']}
        try:
            dsc_metric = json_array[i]['dsc_metric']['dsc_metric']
        except:
            try:
                dsc_metric = json_array[i]['overall_metrics']['dsc_metric']
            except:
                try:
                    dsc_metric = json_array[i]['metrics']['dsc_metric']
                except:
                    try:
                        dsc_metric = json_array[i]['dsc_metric']
                    except:
                        print(f"Error at index {i} for dsc_metric")
                        dsc_metric = None
        data_for_json['dsc_metric'] = dsc_metric

        try:
            nsd_metric = json_array[i]['nsd_metric']['nsd_metric']
        except:
            try:
                nsd_metric = json_array[i]['overall_metrics']['nsd_metric']
            except:
                try:
                    nsd_metric = json_array[i]['metrics']['nsd_metric']
                except:
                    try:
                        nsd_metric = json_array[i]['nsd_metric']
                    except:
                        print(f"Error at index {i} for nsd_metric")
                        nsd_metric = None
        data_for_json['nsd_metric'] = nsd_metric

        new_json.append(data_for_json)
    return new_json

def extract_top_k_preprocessing_functions_to_json(k, json_path, segmenter, test_data_path):
    new_json = get_new_json(json_path)
    top_fns = find_top_k(new_json, lambda x: x['dsc_metric'] + x['nsd_metric'], k)  # Top 3

    result_entries = []

    for i, json_dict in enumerate(top_fns):
        print(f"Looping thru {i + 1} of {len(top_fns)} functions")
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            imgs, boxes, masks = segmenter.loadData(test_data_path)
            spliced_imgs = imgs
            spliced_boxes = boxes
            spliced_masks = masks
            images = ImageData(
                raw=spliced_imgs,
                batch_size=min(8, len(spliced_imgs)),
                image_ids=[i for i in range(len(spliced_imgs))],
                masks=spliced_masks,
                predicted_masks=spliced_masks,
            )
            fn_str = json_dict['preprocessing_function']
            preprocessing_fn = convert_string_to_function(fn_str, 'preprocess_images')
            images = preprocessing_fn(images)
            pred_masks = segmenter.predict(images, spliced_boxes, used_for_baseline=False)
            segmenter.evaluate(pred_masks, spliced_masks)

        stdout_output = stdout_capture.getvalue()
        agent_dsc, agent_nsd = None, None
        for line in stdout_output.splitlines():
            if "Average DSC metric" in line:
                agent_dsc = float(line.split(": ")[1])
            elif "Average NSD metric" in line:
                agent_nsd = float(line.split(": ")[1])

        combined_test = agent_dsc + agent_nsd if agent_dsc and agent_nsd else None
        combined_val = json_dict['dsc_metric'] + json_dict['nsd_metric'] if json_dict['dsc_metric'] and json_dict['nsd_metric'] else None

        entry = {
            "rank": i + 1,
            "preprocessing_function": fn_str,
            "combined_test": combined_test,
            "combined_val": combined_val
        }
        result_entries.append(entry)

    return result_entries

def plot_scatterplot(output_json_path, baseline_val, baseline_test, scatter_output_path):
    x_vals = []
    y_vals = []
    with open(output_json_path, 'r') as f:
        functions = json.load(f)

    for func in functions:
        val_adv = func['combined_val'] - baseline_val
        test_adv = func['combined_test'] - baseline_test
        x_vals.append(val_adv)
        y_vals.append(test_adv)
    

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, alpha=0.7)

    plt.xlabel('Advantage over Validation Baseline')
    plt.ylabel('Advantage over Test Baseline')
    plt.title('Scatterplot: Preprocessing Function Improvements')

    # Add reference lines
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Diagonal reference line: test = val
    grid_size = max(abs(min(x_vals)), abs(max(x_vals)), abs(min(y_vals)), abs(max(y_vals))) + 0.01
    plt.plot([-grid_size, grid_size], [-grid_size, grid_size], 'r--', label='Val = Test')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Axis limits
    plt.xlim([-grid_size, grid_size])
    plt.ylim([-grid_size, grid_size])

    # Save the figure
    plt.savefig(scatter_output_path)
    print(f"Saved scatterplot to {scatter_output_path}")

def plot_aggregated(json_saved_directories, baselines, k, scatter_output_path):
    """
    Aggregates and plots the top k functions from each timestamp subfolder within the provided directories.
    Points from different directories are colored differently and labeled by their folder names.
    """
    print(json_saved_directories)
    plt.figure(figsize=(10, 10))

    # Color palette
    colors = plt.cm.get_cmap('tab10', len(json_saved_directories))

    for idx, (json_path, (baseline_val, baseline_test)) in enumerate(zip(json_saved_directories, baselines)):
        with open(json_path, 'r') as f:
            functions = json.load(f)

        x_vals = []
        y_vals = []

        for func in functions:
            val_adv = func['combined_val'] - baseline_val
            test_adv = func['combined_test'] - baseline_test
            x_vals.append(val_adv)
            y_vals.append(test_adv)

        folder_name = os.path.basename(os.path.dirname(json_path))
        plt.scatter(x_vals, y_vals, alpha=0.7, label=folder_name, color=colors(idx))

    plt.xlabel('Advantage over Validation Baseline')
    plt.ylabel('Advantage over Test Baseline')
    plt.title(f'Scatterplot: Preprocessing Function Improvements with K={k}')

    # Add reference lines
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Diagonal reference line: test = val
    all_vals = x_vals + y_vals  # last ones from loop, just to get bounds
    grid_size = max(abs(min(all_vals)), abs(max(all_vals))) + 0.01
    plt.plot([-grid_size, grid_size], [-grid_size, grid_size], 'r--', label='Val = Test')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Axis limits
    plt.xlim([-grid_size, grid_size])
    plt.ylim([-grid_size, grid_size])

    # Save the figure
    plt.savefig(scatter_output_path)
    print(f"Saved aggregated scatterplot to {scatter_output_path}")

def plot_bar_graph(json_path, baseline_val, baseline_test, bar_output_path):
    plt.figure()
    with open(json_path, 'r') as f:
        functions = json.load(f)
        our_test = functions[0]['combined_test']
    plt.bar('Baseline Test Set', baseline_test, color='b', label=f'Baseline Test Set: {baseline_test:.4f}')
    plt.bar('Our Function Test Set', our_test, color='r', label=f'Our Test Set: {our_test:.4f}')
    
    plt.ylabel('DSC + NSD')
    plt.legend(loc='lower right')
    plt.savefig(bar_output_path)
    print(f"Saved bar graph to {bar_output_path}")

def plot_line_graph(output_path, new_json, combined_metric):
    """ Plot Combined Metric in a single figure """
    combined_metric_lambda = lambda x: x['dsc_metric'] + x['nsd_metric']
    metrics_combined = find_all_metrics(new_json, combined_metric_lambda)
    rolling_highest_combined = find_rolling_highest(new_json, combined_metric_lambda)

    plt.figure(figsize=(10, 5))

    # Plot Combined Metric
    plt.plot(metrics_combined, marker='o', linestyle='-', color='b', label='Eval Combined')
    plt.plot(rolling_highest_combined, marker='x', linestyle='--', color='r', label=f'Rolling Highest Combined: {max(rolling_highest_combined)}')
    plt.axhline(combined_metric, color='g', linestyle='--', label=f'Baseline Combined: {combined_metric}')
    plt.xlabel('Iteration')
    plt.ylabel('Combined Metric')
    plt.title('Combined Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved line graph to {output_path}")

def main(json_path, k, modality, gpu_id):
    # output paths
    timestamp = os.path.basename(os.path.dirname(json_path))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(json_path), f'../{timestamp}'))
    top_k_json_output_path = os.path.abspath(os.path.join(output_dir, f'../../{timestamp}_top_k_output.json'))
    top_1_json_output_path = os.path.abspath(os.path.join(output_dir, f'../../{timestamp}_top_1_output.json'))
    bar_output_path = os.path.abspath(os.path.join(output_dir, f'../../{timestamp}_bar_plot.png'))
    line_graph_output_path = os.path.abspath(os.path.join(output_dir, f'../../{timestamp}_line_graph.png'))
    scatter_output_path = os.path.abspath(os.path.join(output_dir, f'../../{timestamp}_scatter_plot.png'))

    test_data_path = os.path.join(_PROJECT_ROOT, f"data/resized_{modality}_test.pkl")
    baseline_json = os.path.join(_PROJECT_ROOT, f"scratch/{modality}_baseline.json")

    with open(baseline_json, 'r') as f:
        json_array = json.load(f)
        val_baseline = json_array['expert_baseline_val_avg_metric']
        test_baseline = json_array['expert_baseline_test_avg_metric']
        print("val_baseline", val_baseline)
        print("test_baseline", test_baseline)

    segmenter = MedSAMTool(gpu_id=gpu_id, checkpoint_path=os.path.join(_PROJECT_ROOT, "data/medsam_vit_b.pth"))

    print("Extracting top k functions...")
    results_k = extract_top_k_preprocessing_functions_to_json(k, json_path, segmenter, test_data_path)
    with open(top_k_json_output_path, 'w') as f:
        json.dump(results_k, f, indent=4)
    
    print("Extracting top 1 function...")
    results_1 = extract_top_k_preprocessing_functions_to_json(1, json_path, segmenter, test_data_path)
    with open(top_1_json_output_path, 'w') as f:
        json.dump(results_1, f, indent=4)

    plot_line_graph(line_graph_output_path, get_new_json(json_path), val_baseline)
    plot_scatterplot(top_k_json_output_path, val_baseline, test_baseline, scatter_output_path)
    plot_bar_graph(top_1_json_output_path, val_baseline, test_baseline, bar_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots to analyze trajectory of MedSAM rollout.')
    parser.add_argument('--json_path', type=str, required=True, help='Preprocessing func bank path.')
    parser.add_argument('--k', type=int, default=10, help='Number of top functions to extract.')
    parser.add_argument('--modality', type=str, default='dermoscopy', help='Modality to use (e.g., dermoscopy, xray).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for MedSAM.')

    args = parser.parse_args()
    json_path = args.json_path
    k = args.k
    modality = args.modality
    gpu_id = args.gpu_id
    main(json_path, k, modality, gpu_id)
