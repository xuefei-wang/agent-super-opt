import json
import numpy as np
from typing import List, Dict, Callable
import argparse
import matplotlib.pyplot as plt

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

def dump_functions_to_txt(json_array: List[Dict], metric_lambda: Callable[[Dict], float], output_path: str):
    '''Print preprocessing functions and their metric values to a text file for readability'''
    with open(output_path, 'w') as file:
        for obj in json_array:
            metric_value = metric_lambda(obj)
            file.write('\n')
            file.write(f'Value: {metric_value}\n')
            file.write(obj['preprocessing_function'])
            file.write('\n')


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--json_path', type=str, default="output/preprocessing_func_bank.json", help='Path to the JSON file containing the function bank.')
    parser.add_argument('--output_file', type=str, default="figs/functions.txt", help='Path to the output text file.')

    args = parser.parse_args()
    json_path = args.json_path
    output_file = args.output_file

    # Lambda function to map iterations to values for comparison, of type  (Obj) -> Float
    metric_lambda = None # Eg. lambda obj: obj['class_loss'] + obj['regress_loss']

    with open(json_path, 'r') as file:
        json_array = json.load(file)

        # Extract the metric values
        losses = find_all_metrics(json_array, metric_lambda)

        # Find the lowest metric value
        lowest_metric_obj = find_lowest(json_array, metric_lambda)
        rolling_lowest = find_rolling_lowest(json_array, metric_lambda)

        # Pretty print best preprocessing function to console
        print("Best preprocessing function:")
        print(lowest_metric_obj['preprocessing_function'])

        # Write all preprocessing functions and their metrics to a text file
        dump_functions_to_txt(json_array, metric_lambda, output_file)

        # Metrics plot
        plt.plot(losses, marker='o', linestyle='-', color='b', label='Eval metric')
        plt.plot(rolling_lowest, marker='x', linestyle='--', color='r', label='Rolling lowest metric')
        plt.xlabel('Iteration')
        plt.ylabel('Metric')
        plt.title('Metrics during Agent Search')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()