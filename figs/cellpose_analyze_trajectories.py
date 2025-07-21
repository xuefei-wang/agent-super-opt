import os 
import sys
import json
import numpy as np
from typing import List, Dict, Callable
import argparse
import matplotlib.pyplot as plt
# from src.cellpose_segmentation import CellposeTool
# from src.data_io import ImageData
import cv2 as cv
import logging
import sys
import pickle

from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob

# try:
#     from data_io import ImageData
# except ImportError:
#     from src.data_io import ImageData

# try:
#     from tools import BaseSegmenter
# except ImportError:
#     from src.tools import BaseSegmenter

# try:
#     from utils import set_gpu_device
# except ImportError:
#     from src.utils import set_gpu_device

# Dynamically add the project root to sys.path
# This allows Python to find the 'src' module correctly
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Project-specific imports from 'src'
from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
from src.tools import BaseSegmenter # Added direct import
from src.utils import set_gpu_device # Added direct import

from cellpose import models, denoise
from cellpose.io import imread
from cellpose.metrics import average_precision, mask_ious, boundary_scores, aggregated_jaccard_index, flow_error

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


class PriviligedCellposeTool(BaseSegmenter):
    """
    PriviligedCellposeTool is a class that provides a simple interface for the Cellpose model. 
    """
    def __init__(self, model_name: str = "cyto3", device: int = 0, channels: List[int] = [2,1], to_normalize: bool = False, model_kwargs: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.to_normalize = to_normalize

        if device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # R: cytoplasm, G: nucleus
        self.channels = channels

        assert self.model_name in ["cyto3", "denoise_cyto3"], f"Model name {self.model_name} not recognized"

        if self.model_name == "cyto3":
            self.segmenter = models.Cellpose(model_type='cyto3',device=self.device, gpu=True)
        elif self.model_name == "denoise_cyto3":
            self.segmenter = denoise.CellposeDenoiseModel(model_type='cyto3', restore_type='denoise_cyto3', device=self.device, gpu=True)

    def predict(self, images: ImageData, batch_size: int = 8) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[np.ndarray], Any]:
        """
        Predict masks for a batch of images. 
        Args:
            images: ImageData object containing a batch of images. Contains 'raw' and 'masks' attributes in the format of standard ImageData object 
            [batch_size, height, width, channels]. Images provided must be in the format of standard ImageData object and must have two channels, the first channel being the cytoplasm and the second channel being the nucleus.
            batch_size: batch size for prediction
        Returns: 
            masks (List[np.ndarray]): List of labelled images (numpy arrays), where 0=no masks; 1,2,...=mask labels for all pixels in the image
        """

        # Old returns:
        # flows (List[List[np.ndarray]]): List of flow outputs per image:
        # flows[k][0] = XY flow in HSV 0-255
        # flows[k][1] = XY(Z) flows at each pixel
        # flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
        # flows[k][3] = final pixel locations after Euler integration
        # styles (List[np.ndarray]): List of style vectors (size 256) summarizing each image
        # extra (Any): Diameters if using Cellpose model, input images if using denoise model
        to_normalize = self.to_normalize
        raw_list=images.raw
        if self.model_name == "cyto3":
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
        elif self.model_name == "denoise_cyto3":
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
              
        return masks#, flows, styles, extra
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], precision_index: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the performance of the model.
        Args:
            pred_masks: predicted masks
            gt_masks: ground truth masks
        Returns:
            metrics: dictionary of metrics. Contains average_precision at IoU thresholds [0.5, 0.75, 0.9]
            losses: dictionary of losses. Contains bce_loss
        """
        metrics = {}
        losses = {}
        ap, tp, fp, fn  = average_precision(pred_masks, gt_masks)
        # metrics["average_precision"] = ap.mean(axis=0) # Average over all images
        metrics["average_precision"] = np.nanmean(ap, axis=0) # Average over all images
        # metrics["aggregated_jaccard_index"] = aggregated_jaccard_index(pred_masks, gt_masks)
        # metrics["flow_error"] = flow_error(pred_masks, gt_masks)
        # classification loss

        spatial_shape = pred_masks[0].shape[0:2]
        for x in pred_masks:
            if x.shape[0:2] != spatial_shape:
                different_spatial_shapes = True
                break
            else:
                different_spatial_shapes = False

        if not different_spatial_shapes:
            criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
            loss2 = criterion2(torch.Tensor(np.array(pred_masks)), torch.from_numpy(np.array(gt_masks )> 0.5).squeeze().float())
            losses["bce_loss"] = loss2.item()#/len(pred_masks)
        else:
            Copycriterion2 = nn.BCEWithLogitsLoss(reduction="none")
            total_loss = 0
            for pred, gt in zip(pred_masks, gt_masks):
                # Make sure both are tensors with matching dimensions
                pred_tensor = torch.Tensor(pred).view(-1)  # Flatten
                gt_tensor = torch.from_numpy(gt > 0.5).float().view(-1)  # Flatten
                
                # Compute loss for this pair
                mask_loss = Copycriterion2(pred_tensor, gt_tensor).mean()
                total_loss += mask_loss.item()

            losses["bce_loss"] = total_loss / len(pred_masks)
        
        # Let's simplify and only return average precision at [0.5]
        return {"average_precision": metrics["average_precision"][precision_index].item()}

       # return metrics, losses

    def preprocess(self, image_data: ImageData) -> ImageData:
        """We don't need to preprocess the images for Cellpose"""
        return image_data

    def loadData(self, data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load the data from the data path and return a tuple of lists of raw images and gt masks, each as numpy arrays"""
        max_val = 65535 # 16-bit images
        files = sorted(glob.glob(data_path + '*_img.png'))
        raw_images = [(imread(f)).astype(np.float32)/max_val for f in files]
        gt_masks = [imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files]
        gt_masks = [np.expand_dims(mask, axis=2) for mask in gt_masks]
        return raw_images, gt_masks
    
    def loadCombinedDataset(self, data_path: str, dataset_size: int = 256) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Used with combined datasets."""
        # Load all images and masks
        file = glob.glob(os.path.join(data_path, '*'))
        with open(file[0], 'rb') as f:
            data = pickle.load(f)
        images = data['images'][:dataset_size]
        masks = data['masks'][:dataset_size]
        image_ids = data['image_ids'][:dataset_size]
        return images, masks, image_ids 
    
    def evaluateDisaggregated(self, imageData_obj: ImageData, avg_precision_idx: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate the performance of the model on a disaggregated dataset"""
        metrics = {}
        losses = {}
        pred_masks = imageData_obj.predicted_masks
        gt_masks = imageData_obj.masks
        ap, tp, fp, fn  = average_precision(pred_masks, gt_masks)

        img_source_ids = np.array(imageData_obj.image_ids) 
        metrics = {'average_precision': np.nanmean(ap, axis=0)[0].item()}

        bool_mask = img_source_ids == 'cellpose'
        cp_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'bact_phase'
        bp_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'bact_fluor'
        bf_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'tissuenet'
        tn_only_ap = ap[bool_mask]  


        per_dataset = {
            'cellpose': np.nanmean(cp_only_ap, axis=0)[avg_precision_idx].item(),
            'bact_phase': np.nanmean(bp_only_ap, axis=0)[avg_precision_idx].item(),
            'bact_fluor': np.nanmean(bf_only_ap, axis=0)[avg_precision_idx].item(),
            'tissuenet': np.nanmean(tn_only_ap, axis=0)[avg_precision_idx].item()
        }
        
        metrics['disaggregated_average_precision'] = {}
        for name, data in per_dataset.items():
            mean_result = np.nanmean(data, axis=0)
            value = mean_result 
            metrics['disaggregated_average_precision'][name] = None if np.isnan(value) else float(value)

        return metrics
    
def convert_string_to_function(func_str, func_name):
    # Create a namespace dictionary to store the function
    namespace = {}

    # Execute the function string in this namespace
    exec(func_str, globals(), namespace)

    # Return the function object from the namespace
    return namespace[func_name]

    

def main(json_path: str, data_path: str, output_dir: str, precision_index: int = 0, device: int = 0, dataset_size: int = 100, batch_size: int = 16, k: int = 10):
    # Let's save these results into a new subfolder of output_dir
    output_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as file:
        json_array = json.load(file)

    # First evaluate the baseline of no preprocessing
    segmenter = CellposeTool(model_name="cyto3", device=device)
    # raw_images, gt_masks = segmenter.loadData(os.path.join(data_path, 'val_set/'))
    raw_images, gt_masks, image_sources = segmenter.loadCombinedDataset(os.path.join(data_path, 'val_set/'), dataset_size=dataset_size)

    raw_images, gt_masks = raw_images, gt_masks
    images = ImageData(raw=raw_images, masks=gt_masks, batch_size=batch_size, image_ids=image_sources)

    images.predicted_masks = segmenter.predict(images, batch_size=images.batch_size)
    metrics_val_no_preprocessing = segmenter.evaluateDisaggregated(images)
    no_preprocess = metrics_val_no_preprocessing
    print('Evaluated no preprocessing baseline on val (black line)')

    # Use priviliged CellposeTool to get "best" results, to_normalize=True
    priviliged_segmenter = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)

    # raw_images_val, gt_masks_val = priviliged_segmenter.loadData(os.path.join(data_path, 'val_set/'))
    raw_images_val, gt_masks_val, image_sources_val = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'val_set/'), dataset_size=dataset_size)

    images_val = ImageData(raw=raw_images_val, masks=gt_masks_val, batch_size=batch_size, image_ids=image_sources_val)
    images_val.predicted_masks = priviliged_segmenter.predict(images_val, batch_size=images_val.batch_size)
    metrics_val_expert_baseline = priviliged_segmenter.evaluateDisaggregated(images_val)
    print('Evaluated expert baseline on val (cyan line)')

    # raw_images_test, gt_masks_test = priviliged_segmenter.loadData(os.path.join(data_path, 'test_set/'))
    # images = ImageData(raw=raw_images_test, batch_size=8, image_ids=[i for i in range(len(raw_images_test))])
    # pred_masks_test = priviliged_segmenter.predict(images, batch_size=8)
    # metrics_test = priviliged_segmenter.evaluate(pred_masks_test, gt_masks_test)

    # handle json ambiguities
    new_json = []
    for i in range(len(json_array)):
        data_for_json = {'preprocessing_function' : json_array[i]['preprocessing_function']}
        try:
            avg_prec = json_array[i]['average_precision']['average_precision']
        except:
            try: 
                avg_prec = json_array[i]['overall_metrics']['average_precision']
            except:
                try: 
                    avg_prec = json_array[i]['metrics']['average_precision']
                except:
                    try:
                        avg_prec = json_array[i]['average_precision']
                    except:
                        print(f"Error at index {i}")
        data_for_json['average_precision'] = avg_prec
        data_for_json['disaggregated_average_precision'] = json_array[i]['overall_metrics']['disaggregated_average_precision']
        new_json.append(data_for_json) 


    # Plot results for val set
    json_array = new_json
    metric_lambda = lambda obj: obj['average_precision']

    metrics = find_all_metrics(json_array, metric_lambda)
    highest_metric_obj   = find_highest(json_array, metric_lambda)
    rolling_highest = find_rolling_highest(json_array, metric_lambda)

    plt.figure()
    plt.plot(metrics, marker='o', linestyle='-', color='b', label='Eval metric')
    plt.plot(rolling_highest, marker='x', linestyle='--', color='r', label=f'Rolling highest metric: {max(rolling_highest)}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Precision @ IoU=0.5')
    plt.title(f'Validation Metrics during Agent Search: {json_path.split("/")[-2]}')

    plt.axhline(no_preprocess['average_precision'], color='k', linestyle='--', label=f'No preprocessing. IOU threshold 0.5: {no_preprocess["average_precision"]}')
    plt.axhline(metrics_val_expert_baseline['average_precision'], color='cyan', linestyle='--', label=f"Baseline: cellpose's min-max norm {metrics_val_expert_baseline['average_precision']}") 
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, 'metrics_during_agent_search.png'))

    # Now begin the image analysis and attempt to convert string to function

    best_preprocessing_function_str = highest_metric_obj['preprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function_str, 'preprocess_images')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_preprocessing_function.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])

    # Now let's get images with and without preprocessing
    # First evaluate the baseline of no preprocessing
    segmenter = CellposeTool(model_name="cyto3", device=device)
    # raw_images, gt_masks = segmenter.loadData(os.path.join(data_path, 'val_set/'))
    # Re-use raw_images and gt_masks from validation set loading earlier
    # raw_images, gt_masks = segmenter.loadCombinedDataset(os.path.join(data_path, 'val_set/'), dataset_size=dataset_size)

    # raw_images, gt_masks = raw_images, gt_masks
    # images = ImageData(raw=raw_images, batch_size=batch_size, image_ids=[i for i in range(len(raw_images))]) # This is images_val

    orig_images_val = ImageData(raw=[np.copy(img_arr) for img_arr in raw_images_val], masks=gt_masks_val, batch_size=images_val.batch_size, image_ids=image_sources_val)
    best_preprocessed_images_val = best_preprocessing_function(ImageData(raw=[np.copy(img_arr) for img_arr in raw_images_val], masks=gt_masks_val, batch_size=images_val.batch_size, image_ids=image_sources_val))
    agent_images_val = best_preprocessed_images_val.raw
    print("Preprocessed validation images with best function")

    plt.figure()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(min(4, len(orig_images_val.raw))): # Ensure we don't go out of bounds
        axes[0, i].imshow(orig_images_val.raw[i])
        axes[0, i].set_title(f'Original Val Image {i+1}')
        axes[1, i].imshow(agent_images_val[i])
        axes[1, i].set_title(f'Agent Val Image {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessed_val_images.png'))

    # Now let's do test set evaluation

    # # First, the baseline (to_normalize=True)
    # priviliged_segmenter is already initialized
    # raw_images_test, gt_masks_test = priviliged_segmenter.loadData(os.path.join(data_path, 'test_set/'))
    test_dataset_size = 808#08 # hardcoded  
    raw_images_test, gt_masks_test, image_sources_test = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)
    images_test = ImageData(raw=raw_images_test, masks=gt_masks_test, batch_size=batch_size, image_ids=image_sources_test)
    images_test.predicted_masks = priviliged_segmenter.predict(images_test, batch_size=images_test.batch_size)
    metrics_test_expert_baseline = priviliged_segmenter.evaluateDisaggregated(images_test)
    print("Evaluated expert baseline on test (blue bar)")

    # Now let's do the other baseline (to_normalize=False) for test set
    non_privileged_segmenter_test = CellposeTool(model_name="cyto3", device=device) # ensure fresh segmenter for test
    # We use images_test which contains the raw test images
    raw_images_test_no_norm, gt_masks_test_no_norm, image_sources_test_no_norm = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)
    images_test_no_norm = ImageData(raw=raw_images_test_no_norm, masks=gt_masks_test_no_norm, batch_size=batch_size, image_ids=image_sources_test_no_norm)
    images_test_no_norm.predicted_masks = non_privileged_segmenter_test.predict(images_test_no_norm, batch_size=images_test_no_norm.batch_size)
    metrics_test_baseline_no_norm = non_privileged_segmenter_test.evaluateDisaggregated(images_test_no_norm)
    print("Evaluated no preprocessing baseline on test (black bar)")

    # Agent's best function on test set, without to_normalize
    raw_images_test_agent, gt_masks_test_agent, image_sources_test_agent = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)
    images_test_agent = ImageData(raw=raw_images_test_agent, masks=gt_masks_test_agent, batch_size=batch_size, image_ids=image_sources_test_agent)
    best_preprocessed_images_test = best_preprocessing_function(images_test_agent)
    # non_privileged_segmenter_test is already initialized
    images_test_agent.predicted_masks = non_privileged_segmenter_test.predict(best_preprocessed_images_test, batch_size=batch_size)
    metrics_test_agent_function = non_privileged_segmenter_test.evaluateDisaggregated(images_test_agent)
    print("Evaluated top performing agent function on test (red bar)")


    plt.figure()
    plt.bar('expert_baseline', metrics_test_expert_baseline['average_precision'], color='b', label=f"Expert Baseline (min-max norm) Test Set: {metrics_test_expert_baseline['average_precision']:.4f}")
    plt.bar('agent_designed', metrics_test_agent_function['average_precision'], color='r', label=f"Agent Designed Test Set: {metrics_test_agent_function['average_precision']:.4f}")
    plt.bar('no_preprocessing_baseline', metrics_test_baseline_no_norm['average_precision'], color='k', label=f"No Preprocessing Baseline Test Set: {metrics_test_baseline_no_norm['average_precision']:.4f}")
    plt.xlabel('Method')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    plt.title(f'Test Set Performance Comparison: {json_path.split("/")[-2]}')
    plt.savefig(os.path.join(output_dir, 'test_set_metrics.png'))

    # Save expert baseline performances to a JSON file
    expert_baseline_performances = {
        "expert_baseline_val_avg_precision": metrics_val_expert_baseline['average_precision'],
        "expert_baseline_test_avg_precision": metrics_test_expert_baseline['average_precision'],
        "disaggregated_expert_baseline_test_avg_precision": metrics_test_expert_baseline['disaggregated_average_precision'],
        'disaggregated_expert_baseline_val_avg_precision': metrics_val_expert_baseline['disaggregated_average_precision']
    }

    # no_nan_disaggregated_expert_baseline_test_avg_precision = {}
    # for name, data in metrics_test_expert_baseline['disaggregated_average_precision'].items():
    #     if data is None:
    #         no_nan_disaggregated_expert_baseline_test_avg_precision[name] = None
    #     elif isinstance(data, (int, float)):
    #         no_nan_disaggregated_expert_baseline_test_avg_precision[name] = None if np.isnan(data) else float(data)
    #     else:
    #         no_nan_disaggregated_expert_baseline_test_avg_precision[name] = data
    
    # no_nan_disaggregated_expert_baseline_val_avg_precision = {}
    # for name, data in metrics_val_expert_baseline['disaggregated_average_precision'].items():
    #     if data is None:
    #         no_nan_disaggregated_expert_baseline_val_avg_precision[name] = None
    #     elif isinstance(data, (int, float)):
    #         no_nan_disaggregated_expert_baseline_val_avg_precision[name] = None if np.isnan(data) else float(data)
    #     else:
    #         no_nan_disaggregated_expert_baseline_val_avg_precision[name] = data

    # expert_baseline_performances['no_nan_disaggregated_expert_baseline_test_avg_precision'] = no_nan_disaggregated_expert_baseline_test_avg_precision
    # expert_baseline_performances['no_nan_disaggregated_expert_baseline_val_avg_precision'] = no_nan_disaggregated_expert_baseline_val_avg_precision

    with open(os.path.join(output_dir, 'expert_baseline_performances.json'), 'w') as file:
        json.dump(expert_baseline_performances, file, indent=4)
    print(f"Saved expert baseline performances to {os.path.join(output_dir, 'expert_baseline_performances.json')}")

    # Let's find the top performing k functions, evaluate them on the test set and save the results into a new file (pickle)
    def find_top_k(json_array: List[Dict], metric_lambda: Callable[[Dict], float], k: int) -> List[Dict]:
        '''Returns object containing the top k highest metric values from a list of JSON objects.'''
        return sorted(json_array, key=metric_lambda, reverse=True)[:k]
    
    top_k_functions = find_top_k(json_array, metric_lambda, k)

    # Now let's evaluate the top k functions on the test set
    top_k_functions_results_val = []
    top_k_functions_results_test = []
    top_k_functions_str = []
    top_k_functions_disaggregated_val = []
    top_k_functions_disaggregated_test = []

    # priviliged_segmenter_for_top_k = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)
    # Use non_privileged_segmenter_test for evaluating agent functions, as per earlier logic for "Our function, without to_normalize"
    segmenter_for_top_k_agent_eval = CellposeTool(model_name="cyto3", device=device)


    for idx, function_item in enumerate(top_k_functions):
        print(f"Evaluating {idx+1} of top {len(top_k_functions)}")
        current_function_str = function_item['preprocessing_function']
        current_metrics_val_float = function_item['average_precision']
        current_metrics_val_dict = function_item['disaggregated_average_precision']
        if current_function_str == best_preprocessing_function_str: # Compare strings to avoid issues with function object comparison
            current_metrics_test_dict = metrics_test_agent_function # Use already computed result for the best function
            function_str_to_save = current_function_str
        else:
            cur_preprocessing_fn_obj = convert_string_to_function(current_function_str, 'preprocess_images')
            
            # Ensure fresh ImageData object for each preprocessing
            raw_images_test, gt_masks_test, image_sources_test = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)
            images_test = ImageData(raw=raw_images_test, masks=gt_masks_test, batch_size=batch_size, image_ids=image_sources_test)
            current_images_to_preprocess_test = cur_preprocessing_fn_obj(images_test)

            cur_preprocessed_images_test = cur_preprocessing_fn_obj(current_images_to_preprocess_test)
            # Evaluate with non_privileged_segmenter, same as the best agent function
            images_test.predicted_masks = segmenter_for_top_k_agent_eval.predict(cur_preprocessed_images_test, batch_size=batch_size)
            current_metrics_test_dict = segmenter_for_top_k_agent_eval.evaluateDisaggregated(images_test)
            function_str_to_save = current_function_str
        
        top_k_functions_results_test.append(current_metrics_test_dict)
        top_k_functions_results_val.append(current_metrics_val_float)
        top_k_functions_str.append(function_str_to_save)
        top_k_functions_disaggregated_test.append(current_metrics_test_dict['disaggregated_average_precision'])
        top_k_functions_disaggregated_val.append(current_metrics_val_dict)

    print(f"Evaluated top {k} functions on test set using non-privileged segmenter.")

    # Save the results into a new file (json)
    top_k_functions_results_output = []
    for i in range(len(top_k_functions)):
        rank = i + 1
        preprocessing_function_string = top_k_functions_str[i]
        test_metrics_dict = top_k_functions_results_test[i]
        val_metric_float = top_k_functions_results_val[i]
        val_metrics_dict = top_k_functions_results_val[i]
        test_metrics_dict = top_k_functions_results_test[i]
        top_k_functions_results_output.append({
            "rank": rank,
            "preprocessing_function": preprocessing_function_string,
            "average_precision_test": test_metrics_dict['average_precision'],
            "average_precision_val": val_metric_float,
            "disaggregated_average_precision_test": top_k_functions_disaggregated_test[i],
            "disaggregated_average_precision_val": top_k_functions_disaggregated_val[i]
        })
    
    # Let's convert nans to None for all top_k_functions_results_outputs.  the only possible ones are disaggregated_average_precision_test and disaggregated_average_precision_val
    for k_func_obj in top_k_functions_results_output:
        for key, value in k_func_obj.items():
            if key == 'disaggregated_average_precision_test' or key == 'disaggregated_average_precision_val':
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float) and np.isnan(sub_value):
                        k_func_obj[key][sub_key] = None
    
    
    with open(os.path.join(output_dir, 'top_k_functions_results.json'), 'w') as file:
        json.dump(top_k_functions_results_output, file, indent=4)
    print(f"Saved top-k function results to {os.path.join(output_dir, 'top_k_functions_results.json')}")



    # Calculate relative improvements over baseline expert
    # using top_k_functions_results_output
    # Calculate relative improvements over baseline expert
    expert_baseline_val_avg_precision = metrics_val_expert_baseline['average_precision']
    expert_baseline_test_avg_precision = metrics_test_expert_baseline['average_precision']

    function_improvements = []
    for func_obj in top_k_functions_results_output:
        val_improvement = (func_obj['average_precision_val'] - expert_baseline_val_avg_precision)
        test_improvement = (func_obj['average_precision_test'] - expert_baseline_test_avg_precision)

        # Let's also calculate the improvements in the disaggregated metrics
        val_disaggregated_improvements = {}
        for key in metrics_val_expert_baseline['disaggregated_average_precision'].keys():
            func_val = func_obj['disaggregated_average_precision_val'].get(key)
            baseline_val = metrics_val_expert_baseline['disaggregated_average_precision'].get(key)
            
            if func_val is not None and baseline_val is not None:
                val_disaggregated_improvements[key] = func_val - baseline_val
            else:
                val_disaggregated_improvements[key] = None
        
        test_disaggregated_improvements = {}
        for key in metrics_test_expert_baseline['disaggregated_average_precision'].keys():
            func_val = func_obj['disaggregated_average_precision_test'].get(key)
            baseline_val = metrics_test_expert_baseline['disaggregated_average_precision'].get(key)
            
            if func_val is not None and baseline_val is not None:
                test_disaggregated_improvements[key] = func_val - baseline_val
            else:
                test_disaggregated_improvements[key] = None
        
        # Store the improvements along with function info for sorting
        function_improvements.append({
            'val_improvement': val_improvement,
            'test_improvement': test_improvement,
            'function_name': func_obj.get('function_name', 'unknown'),
            'val_disaggregated_improvements': val_disaggregated_improvements,
            'test_disaggregated_improvements': test_disaggregated_improvements
        })

    # Let's also plot a scatter plot of the relative performance of the top k functions on the test set (Y) axis, val set (x) axis
    



    # Create the scatter plot
    plt.figure(figsize=(10, 10))
        # Extract val_improvement and test_improvement from the list of dicts
    val_improvements = [item['val_improvement'] for item in function_improvements]
    test_improvements = [item['test_improvement'] for item in function_improvements]

    plt.scatter(val_improvements, test_improvements, alpha=0.7)
    # plt.scatter(function_improvements['val_improvement'], function_improvements['test_improvement'], alpha=0.7)
    plt.xlabel('Validation Relative Improvement')
    plt.ylabel('Test Relative Improvement')
    plt.title(f'Top {k} Functions over Expert Baseline')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add thicker lines at x=0 and y=0
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)
    
    # Determine axis limits
    
    
    # Calculate dynamic axis limit based on data
    all_values = val_improvements + test_improvements
    if all_values:
        max_abs_val = max(abs(max(all_values)), abs(min(all_values)))
        # Add some padding (10%)
        axis_limit = max_abs_val * 1.1
        # Ensure minimum axis limit
        if axis_limit < 0.005:
            axis_limit = 0.005
        # Round up to nearest 0.005 for cleaner axis
        axis_limit = np.ceil(axis_limit / 0.005) * 0.005
    else:
        axis_limit = 0.02  # Default if no data
    

    plt.xlim([-axis_limit, axis_limit])
    plt.ylim([-axis_limit, axis_limit])
    plt.gca().set_aspect('equal', adjustable='box')

    # Add a diagonal line to show where val=test
    plt.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', alpha=0.5, 
            label='Val = Test performance')
    plt.legend()

    plt.savefig(os.path.join(output_dir, f'scatter_plot_top_{k}_functions_over_expert_baseline.png'))


    # Let's plot a 2x2 grid of the same scatter plots comparing the performance of the top k functions over the expert baseline in the disaggregated metrics
    # Recall metrics['disaggregated_average_precision'] contains keys: 'cellpose', 'bact_phase', 'bact_fluor', 'tissuenet'. We should plot each modality as a separate subplot
    # Plot scatter grid here    
    # 2x2 grid plot
# Let's plot a 2x2 grid of the same scatter plots comparing the performance of the top k functions over the expert baseline
    plt.figure(figsize=(20, 20))
    datasets = ['cellpose', 'bact_phase', 'bact_fluor', 'tissuenet']

    for idx, dataset_name in enumerate(datasets):
        plt.subplot(2, 2, idx + 1)
        plt.title(f'{dataset_name}: Top {k} Functions over Expert Baseline')
        
        # Extract disaggregated improvements for this specific dataset
        val_improvements_for_dataset = []
        test_improvements_for_dataset = []
        
        for func_improvement in function_improvements:
            val_imp = func_improvement['val_disaggregated_improvements'].get(dataset_name)
            test_imp = func_improvement['test_disaggregated_improvements'].get(dataset_name)
            
            if val_imp is not None and test_imp is not None:
                val_improvements_for_dataset.append(val_imp)
                test_improvements_for_dataset.append(test_imp)
        
        if val_improvements_for_dataset:
            plt.scatter(val_improvements_for_dataset, test_improvements_for_dataset, alpha=0.7)
            
            # Determine axis limits for this specific dataset
            all_values = val_improvements_for_dataset + test_improvements_for_dataset
            max_abs_val = max(abs(max(all_values)), abs(min(all_values)))
            # Add some padding (10%) to the maximum absolute value for better visualization
            axis_limit = max_abs_val * 1.1
            # Ensure minimum axis limit if values are very small
            if axis_limit < 0.005:
                axis_limit = 0.005
            # Round up to nearest 0.005 for cleaner axis
            axis_limit = np.ceil(axis_limit / 0.005) * 0.005
        else:
            axis_limit = 0.02  # Default if no data
        
        # Apply all the formatting from the main scatter plot
        plt.xlabel('Validation Relative Improvement')
        plt.ylabel('Test Relative Improvement')
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add thicker lines at x=0 and y=0
        plt.axhline(0, color='black', linewidth=1.5)
        plt.axvline(0, color='black', linewidth=1.5)
        
        # Set axis limits to make it square - each subplot has its own limits
        plt.xlim([-axis_limit, axis_limit])
        plt.ylim([-axis_limit, axis_limit])
        
        # Make the axes perfectly square
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Add a diagonal line to show where val=test
        plt.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', alpha=0.5, 
                label='Val = Test performance')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_grid_top_{k}_functions_over_expert_baseline_disaggregated.png'))
    # NOW CREATE A NEW FIGURE for the bar charts!
    plt.figure(figsize=(20, 20))  # This was missing!

    # Plot bar charts for test performance
    for idx, dataset_name in enumerate(datasets):
        plt.subplot(2, 2, idx + 1)
        
        # Get values and format them
        expert_val = metrics_test_expert_baseline['disaggregated_average_precision'].get(dataset_name)
        agent_val = metrics_test_agent_function['disaggregated_average_precision'].get(dataset_name)
        baseline_val = metrics_test_baseline_no_norm['disaggregated_average_precision'].get(dataset_name)
        
        # Format labels
        expert_label = f"Expert Baseline: {expert_val:.4f}" if expert_val is not None else "Expert Baseline: N/A"
        agent_label = f"Agent Designed: {agent_val:.4f}" if agent_val is not None else "Agent Designed: N/A"
        baseline_label = f"No Preprocessing: {baseline_val:.4f}" if baseline_val is not None else "No Preprocessing: N/A"
        
        # Use 0 for None values for visualization (or skip them)
        expert_val = expert_val if expert_val is not None else 0
        agent_val = agent_val if agent_val is not None else 0
        baseline_val = baseline_val if baseline_val is not None else 0
        
        plt.bar('expert_baseline', expert_val, color='b', label=expert_label)
        plt.bar('agent_designed', agent_val, color='r', label=agent_label)
        plt.bar('no_preprocessing', baseline_val, color='k', label=baseline_label)
        
        plt.xlabel('Method')
        plt.ylabel('Average Precision')
        plt.title(f'{dataset_name} - Test Set Performance')
        plt.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_set_metrics_disaggregated.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the function bank.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory.')
    parser.add_argument('--precision_index', type=int, required=False, default=0, help='Which average precision index to use. [0.5, 0.75, 0.9].')
    parser.add_argument('--device', type=int, required=False, default=0, help='Which GPU to use.')
    parser.add_argument('--dataset_size', type=int, required=False, default=256, help='Number of dataset size to show in the function bank.')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size for Cellpose.')
    parser.add_argument('--k', type=int, required=False, default=10, help='Number of top performing functions to evaluate on the test set.')
    args = parser.parse_args()
    

    
    output_dir = os.path.dirname(args.json_path)
    json_path = args.json_path
    data_path = args.data_path
    # print(f"data_path: {data_path}")
    # data_path = os.path.dirname(os.path.dirname(data_path))
    print(f"data_path: {data_path}")
    main(json_path, data_path, output_dir, args.precision_index, args.device, args.dataset_size, args.batch_size, args.k)
    