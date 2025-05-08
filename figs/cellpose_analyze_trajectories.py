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
            self.segmenter = models.Cellpose(model_type='cyto3',device=self.device, gpu=True)
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
        elif self.model_name == "denoise_cyto3":
            self.segmenter = denoise.CellposeDenoiseModel(model_type='cyto3', restore_type='denoise_cyto3', device=self.device, gpu=True)
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
        metrics["average_precision"] = ap.mean(axis=0) # Average over all images
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
    
def convert_string_to_function(func_str, func_name):
    # Create a namespace dictionary to store the function
    namespace = {}

    # Execute the function string in this namespace
    exec(func_str, globals(), namespace)

    # Return the function object from the namespace
    return namespace[func_name]

    

def main(json_path: str, data_path: str, output_dir: str, precision_index: int = 0, device: int = 0):
    # Let's save these results into a new subfolder of output_dir
    output_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as file:
        json_array = json.load(file)

    # First evaluate the baseline of no preprocessing
    segmenter = CellposeTool(model_name="cyto3", device=device)
    raw_images, gt_masks = segmenter.loadData(os.path.join(data_path, 'val_set/'))
    raw_images, gt_masks = raw_images, gt_masks
    images = ImageData(raw=raw_images, batch_size=16, image_ids=[i for i in range(len(raw_images))])

    pred_masks = segmenter.predict(images, batch_size=images.batch_size)
    metrics_val = segmenter.evaluate(pred_masks, gt_masks)
    no_preprocess = metrics_val

    # Use priviliged CellposeTool to get "best" results, to_normalize=True
    priviliged_segmenter = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)

    raw_images_val, gt_masks_val = priviliged_segmenter.loadData(os.path.join(data_path, 'val_set/'))
    images = ImageData(raw=raw_images_val, batch_size=16, image_ids=[i for i in range(len(raw_images_val))])
    pred_masks_val = priviliged_segmenter.predict(images, batch_size=16)
    metrics_val = priviliged_segmenter.evaluate(pred_masks_val, gt_masks_val, precision_index=precision_index)

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
    plt.axhline(metrics_val['average_precision'], color='cyan', linestyle='--', label=f"Baseline: cellpose's min-max norm {metrics_val['average_precision']}") 
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, 'metrics_during_agent_search.png'))

    # Now begin the image analysis and attempt to convert string to function

    best_preprocessing_function = highest_metric_obj['preprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function, 'preprocess_images')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_preprocessing_function.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])

    # Now let's get images with and without preprocessing
    # First evaluate the baseline of no preprocessing
    segmenter = CellposeTool(model_name="cyto3", device=device)
    raw_images, gt_masks = segmenter.loadData(os.path.join(data_path, 'val_set/'))
    raw_images, gt_masks = raw_images, gt_masks
    images = ImageData(raw=raw_images, batch_size=16, image_ids=[i for i in range(len(raw_images))])

    orig_images = images
    best_preprocessed_images = best_preprocessing_function(ImageData(raw=raw_images, batch_size=16, image_ids=[i for i in range(len(raw_images))]))
    agent_images = best_preprocessed_images.raw
    plt.figure()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        axes[0, i].imshow(orig_images.raw[i])
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[1, i].imshow(agent_images[i])
        axes[1, i].set_title(f'Agent Image {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessed_images.png'))

    # Now let's do test set evaluation

    # # First, the baseline (to_normalize=True)
    priviliged_segmenter = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)
    raw_images_test, gt_masks_test = priviliged_segmenter.loadData(os.path.join(data_path, 'test_set/'))
    images = ImageData(raw=raw_images_test, batch_size=8, image_ids=[i for i in range(len(raw_images_test))])
    pred_masks_test = priviliged_segmenter.predict(images, batch_size=8)
    metrics_test_baseline = priviliged_segmenter.evaluate(pred_masks_test, gt_masks_test, precision_index)

    # Now let's do the other baseline (to_normalize=False)
    # no norm baseline
    non_privileged_segmenter = CellposeTool(model_name="cyto3", device=device)
    pred_masks_test_no_norm = non_privileged_segmenter.predict(images, batch_size=8)
    metrics_test_baseline_no_norm = non_privileged_segmenter.evaluate(pred_masks_test_no_norm, gt_masks_test)


    best_preprocessed_images = best_preprocessing_function(images)
    # Our function, without to_normalize
    non_privileged_segmenter = CellposeTool(model_name="cyto3", device=device)
    pred_masks_test = non_privileged_segmenter.predict(best_preprocessed_images, batch_size=8)
    metrics_test_our_function = non_privileged_segmenter.evaluate(pred_masks_test, gt_masks_test)



    plt.figure()
    plt.bar('baseline', metrics_test_baseline['average_precision'], color='b', label=f'Baseline minmax norm Test Set: {metrics_test_baseline["average_precision"]}')
    plt.bar('agent designed', metrics_test_our_function['average_precision'], color='r', label=f'Our function Test Set: {metrics_test_our_function["average_precision"]}')
    plt.bar('baseline no norm', metrics_test_baseline_no_norm['average_precision'], color='k', label=f'Baseline no norm Test Set: {metrics_test_baseline_no_norm["average_precision"]}')
    plt.xlabel('Method')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'test_set_metrics.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the function bank.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory.')
    parser.add_argument('--precision_index', type=int, required=False, default=0, help='Which average precision index to use. [0.5, 0.75, 0.9].')
    parser.add_argument('--device', type=int, required=False, default=0, help='Which GPU to use.')
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.json_path)
    json_path = args.json_path
    data_path = args.data_path
    data_path = os.path.dirname(os.path.dirname(data_path))
    main(json_path, data_path, output_dir, args.precision_index, args.device)
    