import json
from typing import List, Dict, Callable
import os
import numpy as np
from src.cellpose_segmentation import CellposeTool
import cv2 as cv
from src.data_io import ImageData
test_dir = '/home/afarhang/data/updated_cellpose_combined_data/test_set/'
test_dataset_size = 808
# function to analyze llama
segmenter = CellposeTool(model_name='cyto3', device=1)

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

def evaluate_function_on_test_set(function_obj: Dict, test_dir: str, test_dataset_size: int = 808):
    batch_size = 16
    raw_images_test, masks_test, image_ids_test = segmenter.loadCombinedDataset(test_dir, dataset_size=test_dataset_size)
    images_test = ImageData(raw=raw_images_test, masks=masks_test, image_ids=image_ids_test, batch_size=batch_size)
    current_function = convert_string_to_function(function_obj['preprocessing_function'], 'preprocess_images')
    new_image_data_obj =ImageData(raw=[np.copy(img_arr) for img_arr in raw_images_test], masks=masks_test, batch_size=batch_size, image_ids=image_ids_test)
    processed_images_test = current_function(new_image_data_obj)
    images_test.predicted_masks = segmenter.predict(processed_images_test, batch_size=images_test.batch_size)
    metrics = segmenter.evaluateDisaggregated(images_test)
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
# aggregate top k functions from  
exp_condition = 'all_cellpose_data_llama_0.7'
directory_list = [
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180146',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180151',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180156',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180201',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180206',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180211',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180216',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180221',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180226',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180232',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180236',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180241',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180247',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180252',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180258',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180302',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180308',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180313',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180318',
                    '/home/afarhang/git/sci-agent/cellpose_segmentation/20250515-180323'
                    ]
top_k_functions = aggregate_top_k_functions(directory_list, lambda x: x['overall_metrics']['average_precision'], k=10)



# Let's evaluate the top k functions on the test set (aggregated and disaggregated) We want to log to a json file with the structure:
# [{"rank": , "preprocessing_function": ,
#  "average_precision_test": ,
#  "average_precision_val": ,
#  "disaggregated_average_precision_test": {"cellpose": , "bact_phase": , "bact_fluor": , "tissuenet": },
#  "disaggregated_average_precision_val": {"cellpose": , "bact_phase": , "bact_fluor": , "tissuenet": }}, ...]

to_save = []
for idx, func_obj in enumerate(top_k_functions):
    print(f"Evaluating function {idx+1} of {len(top_k_functions)}")
    metrics = evaluate_function_on_test_set(func_obj, test_dir, test_dataset_size)
    func_obj['rank'] = idx
    func_obj['average_precision_test'] = metrics['average_precision']
    func_obj['disaggregated_average_precision_test'] = metrics['disaggregated_average_precision']
    func_obj['average_precision_val'] = func_obj['overall_metrics']['average_precision']
    func_obj['disaggregated_average_precision_val'] = func_obj['overall_metrics']['disaggregated_average_precision']
    to_save.append(func_obj)






# save json here
with open(f'_top_k_functions_{exp_condition}.json', 'w') as f:
    json.dump(to_save, f)
print(f"Saved top_k_functions_{exp_condition}.json")
# [print(x['overall_metrics']['average_precision']) for x in top_k_functions]

