from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob

try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData

try:
    from tools import BaseSegmenter
except ImportError:
    from src.tools import BaseSegmenter

try:
    from utils import set_gpu_device
except ImportError:
    from src.utils import set_gpu_device

from cellpose import models, denoise
from cellpose.io import imread
from cellpose.metrics import average_precision, mask_ious, boundary_scores, aggregated_jaccard_index, flow_error

class CellposeTool(BaseSegmenter):
    """
    CellposeTool is a class that provides a simple interface for the Cellpose model. 
    """
    def __init__(self, model_name: str = "cyto3", device: int = 0, model_kwargs: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs

        if device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # R: cytoplasm, G: nucleus
        self.channels = channels = [2,1]

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
        to_normalize = True
        raw_list=images.raw
        if self.model_name == "cyto3":
            self.segmenter = models.Cellpose(model_type='cyto3',device=self.device, gpu=True)
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
        elif self.model_name == "denoise_cyto3":
            self.segmenter = denoise.CellposeDenoiseModel(model_type='cyto3', restore_type='denoise_cyto3', device=self.device, gpu=True)
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
              
        return masks#, flows, styles, extra
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
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
        return metrics, losses

    def preprocess(self, image_data: ImageData) -> ImageData:
        """We don't need to preprocess the images for Cellpose"""
        return image_data

    def loadData(self, data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load the data from the data path and return a tuple of lists of raw images and gt masks, each as numpy arrays"""
        files = sorted(glob.glob(data_path + '*_img.png'))
        raw_images = [imread(f) for f in files]
        gt_masks = [imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files]
        gt_masks = [np.expand_dims(mask, axis=2) for mask in gt_masks]
        return raw_images, gt_masks
    

if __name__ == "__main__":
    device = 1  
    # set_gpu_device(device)
    cellpose_tool = CellposeTool(model_name="cyto3", device=device)
    num_files = 100
    # files = [f'/home/alexfarhang/data/cellpose/train/{num:03d}_img.png' for num in range(num_files)]
    files = sorted(glob.glob('/home/alexfarhang/data/cellpose/train/*_img.png'))[:num_files]
    # raw_images = np.array([imread(f) for f in files], dtype=object)
    # raw_images = np.array([imread(f) for f in files])
    raw_images = [imread(f) for f in files]
    # transposed_raw = np.transpose(raw_images, axes=[0, 3, 1, 2])
    # gt_masks = [np.expand_dims(np.array(imread((f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]))), axis=2) for f in files]
    # gt_masks = np.array([imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files])
    gt_masks = [imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files]
    gt_masks = [np.expand_dims(mask, axis=2) for mask in gt_masks]
    # gt_masks = np.expand_dims(gt_masks, axis=1)
    # gt_masks = np.expand_dims(gt_masks, axis=3)

    images = ImageData(raw=raw_images,
                       batch_size=8,
                       image_ids=[i for i in range(num_files)],
                       masks=gt_masks,
                       predicted_masks=gt_masks)


    pred_masks, flows, styles, imgs = cellpose_tool.predict(images, batch_size=images.batch_size)
    # gt_masks_list = [images.masks[i] for i in range(images.masks.shape[0])] 
    gt_masks_list = images.masks
    metrics, losses = cellpose_tool.evaluate(pred_masks, gt_masks_list)
    print(metrics)
    print(losses)
    print('done')