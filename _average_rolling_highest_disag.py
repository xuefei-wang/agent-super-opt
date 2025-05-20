import json
import numpy as np
import os
import matplotlib.pyplot as plt

# Fill with directories of experiments
list_of_directories = ['/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011141',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011146',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011151',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011156',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011201',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011206',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011211',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011216',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011221',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011226',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011231',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011237',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011241',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011247',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011252',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011257',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011302',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011307',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011313',
                        '/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011318'
]

output_file = 'mean_std_rolling_best_disaggregated.png'


def analyze_json_files_disaggregated(file_paths):
    """Analyze JSON files and compute rolling maximums for disaggregated metrics."""
    datasets = ['cellpose', 'bact_phase', 'bact_fluor', 'tissuenet']
    
    # Store rolling maximums for each dataset across all rollouts
    all_rollouts = {dataset: [] for dataset in datasets}
    
    for file_path in file_paths:
        with open(os.path.join(file_path, 'preprocessing_func_bank.json'), 'r') as f:
            curr_iter_list = json.load(f)
        
        # Initialize rolling max tracking for this rollout
        rolling_max = {dataset: [] for dataset in datasets}
        current_max = {dataset: float('-inf') for dataset in datasets}
        
        for obj in curr_iter_list:
            disaggregated_metrics = obj['overall_metrics']['disaggregated_average_precision']
            
            # Update rolling max for each dataset
            for dataset in datasets:
                metric_value = disaggregated_metrics.get(dataset)
                if metric_value is not None:
                    current_max[dataset] = max(current_max[dataset], metric_value)
                rolling_max[dataset].append(current_max[dataset])
        
        # Add this rollout's data to all rollouts
        for dataset in datasets:
            all_rollouts[dataset].append(rolling_max[dataset])
    
    # Compute mean and std for each dataset
    results = {}
    for dataset in datasets:
        dataset_rollouts = all_rollouts[dataset]
        
        # Trim to same length
        if dataset_rollouts:
            min_len = min(len(lst) for lst in dataset_rollouts)
            trimmed = [lst[:min_len] for lst in dataset_rollouts]
            trimmed_array = np.array(trimmed)
            
            # Replace -inf with NaN for proper statistics
            trimmed_array[trimmed_array == float('-inf')] = np.nan
            
            # Compute statistics
            results[dataset] = {
                'mean': np.nanmean(trimmed_array, axis=0),
                'std': np.nanstd(trimmed_array, axis=0)
            }
        else:
            results[dataset] = {
                'mean': np.array([]),
                'std': np.array([])
            }
    
    return results


if __name__ == "__main__":
    # Analyze the data
    results = analyze_json_files_disaggregated(list_of_directories)
    
    # Load baseline performances
    with open(os.path.join(list_of_directories[0], 'analysis_results/expert_baseline_performances.json'), 'r') as f:
        reference_expert_baseline_performances = json.load(f)
    
    baseline_val_disaggregated = reference_expert_baseline_performances.get(
        'disaggregated_expert_baseline_val_avg_precision', {}
    )
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = ['cellpose', 'bact_phase', 'bact_fluor', 'tissuenet']
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        
        # Get data for this dataset
        data = results[dataset_name]
        mean = data['mean']
        std = data['std']
        
        if len(mean) > 0:
            # Plot mean and std
            ax.plot(mean, label='Mean Rolling Maximum', color='blue')
            ax.fill_between(range(len(mean)), mean - std, mean + std, 
                           color='blue', alpha=0.2, label='Standard Deviation')
            
            # Add baseline if available
            baseline_val = baseline_val_disaggregated.get(dataset_name)
            if baseline_val is not None:
                ax.axhline(y=baseline_val, color='red', linestyle='--', 
                          label=f'Baseline Val: {baseline_val:.4f}')
            
            # Set y-axis limits based on data
            valid_values = mean[~np.isnan(mean)]
            if len(valid_values) > 0:
                y_min = max(0, np.min(valid_values) - 0.05)
                y_max = np.max(valid_values) + 0.05
                ax.set_ylim(y_min, y_max)
        
        ax.set_title(f'{dataset_name} - Rolling Maximum Scores')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Average Precision')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Mean and Standard Deviation of Rolling Maximum Scores by Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved disaggregated plot to {output_file}")