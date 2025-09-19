import json
import os
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt

def combine_top_k_results(task_name):
    all_functions = []

    for path in os.listdir(task_name):
        if path.startswith('2025'):
            json_path = os.path.join(task_name, path, 'analysis_results', 'top_k_functions_results.json')
            print(json_path)
            
            with open(json_path, 'r') as file:
                json_array = json.load(file)

            json_array_new = copy.deepcopy(json_array)
            for entry in json_array_new:
                entry['expr'] = path

            all_functions += json_array_new


    all_functions.sort(key=lambda x: x.get('average_f1_val', 0), reverse=True)

    for idx, entry in enumerate(all_functions):
        entry['rank'] = idx+1

    with open(os.path.join(task_name, 'top_k_across_all_reps.json'), 'w') as file:
        json.dump(all_functions, file, indent=4)




def plot_learning_curve(task_name):
    if task_name == "cellpose_segmentation":
        obj_func = lambda obj: obj['overall_metrics']['average_precision']
    elif task_name == "medSAM_segmentation":
        obj_func = lambda obj: obj['overall_metrics']['dsc_metric'] + obj['overall_metrics']['nsd_metric']
    elif task_name == "spot_detection":
        obj_func = lambda obj: obj['overall_metrics']['f1_score']

    # ======= get mean/std plot ======
    rolling_best_output_filepath = os.path.join(task_name, 'global_rolling_max.png')

    def analyze_json_files(file_paths):
        rolling_maximums = []

        for file_path in file_paths:
            # Load baseline from the first directory
            with open(os.path.join(file_path, 'preprocessing_func_bank.json'), 'r') as f:
                curr_iter_list = json.load(f)
            
            # Per rollout
            rolling_max = []
            current_max = float('-inf')
            for obj in curr_iter_list:
                combined_score = obj_func(obj)
                current_max = max(current_max, combined_score)
                rolling_max.append(current_max)
            
            # All rollouts
            rolling_maximums.append(rolling_max)
        
        # Trim to same length iteration
        min_len = min(len(lst) for lst in rolling_maximums)
        trimmed_lists = [lst[:min_len] for lst in rolling_maximums]

        # Compute mean and standard deviation of rolling maximums at each index
        trimmed_lists = np.array(trimmed_lists)
        mean_rolling_max = np.mean(trimmed_lists, axis=0)
        std_rolling_max = np.std(trimmed_lists, axis=0)

        return mean_rolling_max, std_rolling_max

    list_of_paths = []
    for path in os.listdir(task_name):
        if path.startswith('2025'):
            list_of_paths.append(os.path.join(task_name, path))

    mean, std = analyze_json_files(list_of_paths)

    if task_name == "cellpose_segmentation":
        with open(os.path.join(list_of_paths[0], "analysis_results", "expert_baseline_performances.json"), 'r') as f:
            baseline_data = json.load(f)
        val_baseline = baseline_data["expert_baseline_val_avg_precision"]
    elif task_name == "medSAM_segmentation":
        with open(os.path.join(task_name, "expert_baseline_performances.json"), 'r') as f:
            baseline_data = json.load(f)
        val_baseline = baseline_data["expert_baseline_val_avg_metric"]
    elif task_name == "spot_detection":
        with open(os.path.join(list_of_paths[0], "analysis_results", "expert_baseline_performances.json"), 'r') as f:
            baseline_data = json.load(f)
        val_baseline = baseline_data["expert_baseline_val_f1"]
    else:
        raise ValueError(f"Unknown task_name: {task_name}")


    plt.figure(figsize=(10, 5))
    plt.plot(mean, label=f'Mean Rolling Maximum ({np.max(mean):.3f})', color='blue')
    plt.fill_between(range(len(mean)), mean - std, mean + std, color='blue', alpha=0.2, label='Standard Deviation')
    plt.axhline(y=val_baseline, color='red', linestyle='--', label=f'Baseline Val ({val_baseline:.3f})')
    plt.title('Mean and Standard Deviation of Rolling Maximum Scores')
    plt.xlabel('Iterations')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid()
    plt.savefig(rolling_best_output_filepath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine top-K results from repetitions and plot learning curves.')
    parser.add_argument('--task_name', type=str, required=True, help='Task name: medSAM_segmentation, cellpose_segmentation, spot_detection')
    
    args = parser.parse_args()
    task_name = args.task_name
    combine_top_k_results(task_name)
    plot_learning_curve(task_name)