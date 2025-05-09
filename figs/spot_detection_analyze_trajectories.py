import os 
import sys
import json
import numpy as np
from typing import List, Dict, Callable
import argparse
import matplotlib.pyplot as plt
import cv2 as cv
import logging
import sys

from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob

# Dynamically add the project root to sys.path
# This allows Python to find the 'src' module correctly
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.spot_detection import DeepcellSpotsDetector
from src.data_io import ImageData



import os
# Analyze a agent search trajectory
# Usage: python figs/fb_analysis.py --json_path <path_to_json> --output_file <output_file>

def find_lowest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns object with the lowest metric value from a list of JSON objects.'''
    return min(json_array, key=metric_lambda)

def find_all_metrics(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> List[float]:
    '''Returns a list of metric values from a list of JSON objects.'''
    return [metric_lambda(obj) for obj in json_array]

def find_rolling_lowest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns a list of metric values, each index being the lowest value up until that point'''
    rolling_lowest = []
    current_lowest = float('inf')
    for obj in json_array:
        metric_value = metric_lambda(obj)
        if metric_value < current_lowest:
            current_lowest = metric_value
        rolling_lowest.append(current_lowest)
    return rolling_lowest

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

    
def convert_string_to_function(func_str, func_name):
    # Create a namespace dictionary to store the function
    namespace = {}

    # Execute the function string in this namespace
    exec(func_str, globals(), namespace)

    # Return the function object from the namespace
    return namespace[func_name]

    

def main(json_path: str, data_path: str, output_dir: str):
    # Let's save these results into a new subfolder of output_dir
    output_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as file:
        json_array = json.load(file)
        
        
    # --- Initialize spot detector tool ---
    deepcell_spot_detector = DeepcellSpotsDetector()
    spots_data = np.load(f"{data_path}", allow_pickle=True)

    # --- Prepare ImageData ---
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])    

    pred = deepcell_spot_detector.predict(images)

    metrics_val = deepcell_spot_detector.evaluate(pred, spots_data['y'])


    # handle json ambiguities
    new_json = []
    for i in range(len(json_array)):
        data_for_json = {'preprocessing_function' : json_array[i]['preprocessing_function']}
        try:
            avg_prec = json_array[i]['f1_score']['f1_score']
        except:
            try: 
                avg_prec = json_array[i]['overall_metrics']['f1_score']
            except:
                try: 
                    avg_prec = json_array[i]['metrics']['f1_score']
                except:
                    try:
                        avg_prec = json_array[i]['f1_score']
                    except:
                        print(f"Error at index {i}")
        data_for_json['f1_score'] = avg_prec
        new_json.append(data_for_json) 


    # Plot results for val set
    json_array = new_json
    metric_lambda = lambda obj: obj['f1_score']

    metrics = find_all_metrics(json_array, metric_lambda)
    highest_metric_obj   = find_highest(json_array, metric_lambda)
    rolling_highest = find_rolling_highest(json_array, metric_lambda)

    plt.figure()
    plt.plot(metrics, marker='o', linestyle='-', color='b', label='Eval metric')
    plt.plot(rolling_highest, marker='x', linestyle='--', color='r', label=f'Rolling highest metric: {max(rolling_highest)}')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Metrics during Agent Search: {json_path.split("/")[-2]}')

    plt.axhline(metrics_val['f1_score'], color='cyan', linestyle='--', label=f"Baseline val F1: {metrics_val['f1_score']}") 
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, 'metrics_during_agent_search.png'))

    # Now begin the image analysis and attempt to convert string to function

    best_preprocessing_function = highest_metric_obj['preprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function, 'preprocess_images')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_preprocessing_function.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])
    

    # Baseline on test
    test_path = os.path.join(os.path.dirname(data_path), 'test.npz')

    spots_data = np.load(f"{test_path}", allow_pickle=True)
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])
    pred = deepcell_spot_detector.predict(images)
    metrics_test_baseline = deepcell_spot_detector.evaluate(pred, spots_data['y'])


    # Agent on test    
    spots_data = np.load(f"{test_path}", allow_pickle=True)

    # --- Prepare ImageData ---
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])    

    best_preprocessed_images = best_preprocessing_function(images)
    pred_test = deepcell_spot_detector.predict(best_preprocessed_images)
    metrics_test_our_function = deepcell_spot_detector.evaluate(pred_test, spots_data['y'])



    plt.figure()
    plt.bar('baseline', metrics_test_baseline['f1_score'], color='b', label=f'Baseline Test Set: {metrics_test_baseline["f1_score"]}')
    plt.bar('agent designed', metrics_test_our_function['f1_score'], color='r', label=f'Our function Test Set: {metrics_test_our_function["f1_score"]}')
    plt.xlabel('Method')
    plt.ylabel('Average F1')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'test_set_metrics.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the function bank.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory.')
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.json_path)
    json_path = args.json_path
    data_path = args.data_path
    main(json_path, data_path, output_dir)
    