import json
import numpy as np
import os
import matplotlib.pyplot as plt

# Fill with directories of experiments
# Vanilla aggregate data
# data_subset = 'all'
# to_plot_disaggregated = False
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
# to_plot_disaggregated = False
    
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



# # Vanilla aggregated experiments with ablations: llama
# data_subset = 'all_data_llama'
# to_plot_disaggregated = False


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


output_file = f'mean_std_rolling_best_{data_subset}.png'



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
            avg_precision = obj['overall_metrics']['average_precision']
            current_max = max(current_max, avg_precision)
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


if __name__ == "__main__":
    mean, std = analyze_json_files(list_of_directories)
    # Plot mean and std deviation range over each iteration
    
    # Load baseline from the first directory
    with open(os.path.join(list_of_directories[0], 'analysis_results/expert_baseline_performances.json'), 'r') as f:
        reference_expert_baseline_performances = json.load(f)
    

    baseline_val = reference_expert_baseline_performances['expert_baseline_val_avg_precision']
    baseline_test = reference_expert_baseline_performances['expert_baseline_test_avg_precision']

    plt.figure(figsize=(10, 5))
    plt.plot(mean, label='Mean Rolling Maximum', color='blue')
    plt.fill_between(range(len(mean)), mean - std, mean + std, color='blue', alpha=0.2, label='Standard Deviation')
    plt.axhline(y=baseline_val, color='red', linestyle='--', label=f'Baseline Val (Test: {baseline_test:.4f})')
    plt.title(f'Mean and Standard Deviation of Rolling Maximum Scores ({data_subset})')
    plt.xlabel('Iterations')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)