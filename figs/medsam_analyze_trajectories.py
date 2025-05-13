import json
import os
import json
from typing import List, Dict, Callable
import numpy as np
import cv2 as cv
import pandas as pd
import argparse
# add src to path
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "..")) # scratch folder
if project_root not in sys.path:
    sys.path.append(project_root)
    
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

def extract_top_k_preprocessing_functions(json_path, exp_timestamp, csv_path):
    new_json = get_new_json(json_path)
    all_fns = find_top_k(new_json, lambda x: x['dsc_metric'] + x['nsd_metric'], 5)  # list of 3 functions

    all_k_metrics = []
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:  # Open in append mode
        csv_writer = csv.writer(csvfile)
        if not file_exists:  # Write header only if file doesn't exist
            csv_writer.writerow(['exp_timestamp', 'preprocessing_function', 'agent_dsc_TEST', 'agent_nsd_TEST', 'agent_combined_TEST', 'agent_dsc_VAL', 'agent_nsd_VAL', 'agent_combined_VAL'])

        for i, json_dict in enumerate(all_fns):
            print("==========================================================================")
            # Capture stdout
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                # Code block where MedSAMTool is usedj
                segmenter = MedSAMTool(gpu_id=3, checkpoint_path="/home/sstiles/sci-agent/data/medsam_vit_b.pth")

                imgs, boxes, masks = segmenter.loadData('/home/sstiles/sci-agent/scratch/resized_dermoscopy_test_filenames_25.pkl')
                used_imgs = imgs
                used_boxes = boxes
                used_masks = masks

                images = ImageData(
                    raw=used_imgs,
                    batch_size=min(8, len(used_imgs)),
                    image_ids=[i for i in range(len(used_imgs))],
                    masks=used_masks,
                    predicted_masks=used_masks,
                )

                fn_str = json_dict['preprocessing_function']
                best_preprocessing_fn = convert_string_to_function(fn_str, 'preprocess_images')
                images = best_preprocessing_fn(images)
                pred_masks = segmenter.predict(images, used_boxes, used_for_baseline=False)
                segmenter.evaluate(pred_masks, used_masks)

            # Extract metrics from captured stdout
            stdout_output = stdout_capture.getvalue()
            print(stdout_output)  # Print captured stdout for progress messages
            lines = stdout_output.splitlines()
            for line in lines:
                if "Average DSC metric" in line:
                    agent_dsc = float(line.split(": ")[1])
                elif "Average NSD metric" in line:
                    agent_nsd = float(line.split(": ")[1])

            print(f"Ran idx {i + 1}/{len(all_fns)}: {agent_dsc + agent_nsd}")

            # Calculate validation metrics
            agent_dsc_val = json_dict['dsc_metric']
            agent_nsd_val = json_dict['nsd_metric']
            # Write the metrics to the CSV file
            csv_writer.writerow([exp_timestamp, fn_str, agent_dsc, agent_nsd, agent_dsc + agent_nsd, agent_dsc_val, agent_nsd_val, agent_dsc_val + agent_nsd_val])
            all_k_metrics.append((fn_str, agent_dsc, agent_nsd))
    '''Returns object with the lowest metric value from a list of JSON objects.'''
    return min(json_array, key=metric_lambda)

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

def plot_combined_fig(fig_output_path, new_json, baseline_dsc_metric, baseline_nsd_metric):
    dsc_metric_lambda = lambda x: x['dsc_metric']
    nsd_metric_lambda = lambda x: x['nsd_metric']
    combined_metric_lambda = lambda x: x['dsc_metric'] + x['nsd_metric']
    
    """ Plot combined DSC, NSD, and Combined metrics in a single figure """
    metrics_dsc = find_all_metrics(new_json, dsc_metric_lambda)
    metrics_nsd = find_all_metrics(new_json, nsd_metric_lambda)
    metrics_combined = find_all_metrics(new_json, combined_metric_lambda)

    rolling_highest_dsc = find_rolling_highest(new_json, dsc_metric_lambda)
    rolling_highest_nsd = find_rolling_highest(new_json, nsd_metric_lambda)
    rolling_highest_combined = find_rolling_highest(new_json, combined_metric_lambda)

    plt.figure(figsize=(15, 5))

    # Plot DSC
    plt.subplot(1, 3, 1)
    plt.plot(metrics_dsc, marker='o', linestyle='-', color='b', label='Eval DSC')
    plt.plot(rolling_highest_dsc, marker='x', linestyle='--', color='r', label=f'Rolling highest DSC: {max(rolling_highest_dsc)}')
    plt.axhline(baseline_dsc_metric, color='g', linestyle='--', label=f'Baseline DSC: {baseline_dsc_metric}')
    plt.xlabel('Iteration')
    plt.ylabel('DSC Metric')
    plt.title('DSC Metrics')
    plt.legend()

    # Plot NSD
    plt.subplot(1, 3, 2)
    plt.plot(metrics_nsd, marker='o', linestyle='-', color='b', label='Eval NSD')
    plt.plot(rolling_highest_nsd, marker='x', linestyle='--', color='r', label=f'Rolling highest NSD: {max(rolling_highest_nsd)}')
    plt.axhline(baseline_nsd_metric, color='g', linestyle='--', label=f'Baseline NSD: {baseline_nsd_metric}')
    plt.xlabel('Iteration')
    plt.ylabel('NSD Metric')
    plt.title('NSD Metrics')
    plt.legend()

    # Plot Combined
    plt.subplot(1, 3, 3)
    plt.plot(metrics_combined, marker='o', linestyle='-', color='b', label='Eval Combined')
    plt.plot(rolling_highest_combined, marker='x', linestyle='--', color='r', label=f'Rolling highest Combined: {max(rolling_highest_combined)}')
    plt.axhline(baseline_dsc_metric + baseline_nsd_metric, color='g', linestyle='--', label=f'Baseline Combined: {baseline_dsc_metric + baseline_nsd_metric}')
    plt.xlabel('Iteration')
    plt.ylabel('Combined Metric')
    plt.title('Combined Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_output_path)
    plt.close()

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

    # scrape the DSC metric from that file
    with open(val_baseline_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Average DSC metric' in line:
                baseline_dsc_metric = float(line.split(': ')[1])
            if 'Average NSD metric' in line:
                baseline_nsd_metric = float(line.split(': ')[1])
    return baseline_dsc_metric, baseline_nsd_metric

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
    print(f"Plot saved to {scatter_output_path}")

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
    print(f"Plot saved to {scatter_output_path}")

def plot_bar(json_path, baseline_val, baseline_test, bar_output_path):
    plt.figure()
    with open(json_path, 'r') as f:
        functions = json.load(f)
        our_test = functions[0]['combined_test']
    plt.bar('Baseline Test Set', baseline_test, color='b', label=f'Baseline Test Set: {baseline_test:.4f}')
    plt.bar('Our Function Test Set', our_test, color='r', label=f'Our Test Set: {our_test:.4f}')
    
    plt.ylabel('DSC + NSD')
    plt.legend(loc='lower right')
    print(f"Saved to {bar_output_path}")
    plt.savefig(bar_output_path)

def main(modality, outer_folder_name, k, start_idx, end_idx):  
    outer_folder_path = f'/home/sstiles/sci-agent/output/{outer_folder_name}/medSAM_segmentation/'
    nontrivial_subfolders = os.listdir(outer_folder_path)[start_idx:end_idx]
    print(f"Looping through timestamp folders: {nontrivial_subfolders}")
    baselines = []
    for timestamp_folder in nontrivial_subfolders:
        test_data_path = f"/home/sstiles/sci-agent/scratch/resized_{modality}_test_filenames_25.pkl"
        val_baseline_dsc, val_baseline_nsd, val_baseline_sum = get_baseline(f"/home/sstiles/sci-agent/scratch/baseline_{modality}_val_filenames_25.txt")
        _, _, test_baseline_sum = get_baseline(f"/home/sstiles/sci-agent/scratch/baseline_{modality}_test_filenames_25.txt")
        baselines.append((val_baseline_sum, test_baseline_sum))

        plot_bar(f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/output.json', val_baseline_sum, test_baseline_sum, f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/bar_plot.png')

        # input_json_path = f'/home/sstiles/sci-agent/output/{outer_folder_name}/medSAM_segmentation/{timestamp_folder}/preprocessing_func_bank.json'
        
        # output_json_path = f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/output.json'
        # if not os.path.exists(os.path.dirname(output_json_path)):
        #     os.makedirs(os.path.dirname(output_json_path))
        # figure_path = f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/combined_metrics.png'
        # if not os.path.exists(os.path.dirname(figure_path)):
        #     os.makedirs(os.path.dirname(figure_path))
        # scatter_output_path = f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/scatter_plot.png'
        # if not os.path.exists(os.path.dirname(scatter_output_path)):
        #     os.makedirs(os.path.dirname(scatter_output_path))

        # plot_combined_fig(figure_path, get_new_json(input_json_path), val_baseline_dsc, val_baseline_nsd)

        # segmenter = MedSAMTool(gpu_id=4, checkpoint_path="/home/sstiles/sci-agent/data/medsam_vit_b.pth")
        # results = extract_top_k_preprocessing_functions_to_json(k, input_json_path, segmenter, test_data_path)
        # print(results)

        # # Ensure the directory exists and has write permissions
        # os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        # with open(output_json_path, 'w') as f:
        #     json.dump(results, f, indent=4)
        # plot_scatterplot(output_json_path, val_baseline_sum, test_baseline_sum, scatter_output_path)

    # json_saved_directories = []
    # for timestamp_folder in nontrivial_subfolders:
    #     json_saved_directories.append(f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}/{k}/{timestamp_folder}/output.json')
    # plot_aggregated(json_saved_directories, baselines, k, f'/home/sstiles/sci-agent/exp_output/{outer_folder_name}_{k}_aggregated_plot.png')

# Removed the erroneous call to main() without arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory for MedSAM.')
    parser.add_argument('--modality', type=str, default='xray', help='Modality to analyze (e.g., dermoscopy, histology).')
    parser.add_argument('--outer_folder_name', type=str, default='2_trial_16_exps', help='Outer folder name for the experiment.')
    parser.add_argument('--k', type=int, required=True, help='Top K preprocessing functions to run the test set on.')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for subfolder iteration.')
    parser.add_argument('--end_idx', type=int, default=8, help='End index for subfolder iteration.')

    args = parser.parse_args()
    modality = args.modality
    outer_folder_name = args.outer_folder_name
    k = args.k
    start_idx = args.start_idx
    end_idx = args.end_idx

    print(f"Running with the following parameters: {modality}, {outer_folder_name}, {k}, {start_idx}, {end_idx}")
    main(modality, outer_folder_name, k, start_idx, end_idx)

# example usage:
# python medsam_analyze_trajectories.py --modality xray --outer_folder_name 2_trial_16_exps --k 1 --start_idx 0 --end_idx 8
# python medsam_analyze_trajectories.py --modality xray --outer_folder_name 2_trial_16_exps --k 3 --start_idx 0 --end_idx 8