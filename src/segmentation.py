"""
Module for cell segmentation model implementations.
Supports both PyTorch and TensorFlow based models through framework-agnostic interfaces.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import replace
import logging

import torch
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label

from deepcell.applications import Mesmer
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from .data_io import ImageData, standardize_mask


def calculate_metrics(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> List[Dict[str, float]]:
    """Calculate segmentation metrics between ground truth and predicted masks.

    This function computes various evaluation metrics to assess the quality of cell
    segmentation predictions against ground truth masks. It handles batched inputs
    and can return either per-batch metrics or averaged metrics across the batch.

    The function uses the Hungarian algorithm to find optimal matching between
    predicted and ground truth objects, then computes various metrics based on
    these matchings. Objects are considered matched only if their IoU exceeds
    the specified threshold.

    Args:
        true_mask: Ground truth segmentation mask with shape (B, 1, H, W) where each
                  unique positive integer represents a distinct object. Background
                  should be labeled as 0.
        pred_mask: Predicted segmentation mask with shape (B, 1, H, W) using the same
                  labeling convention as true_mask.
        iou_threshold: Minimum Intersection over Union (IoU) required to consider an
                      object as correctly detected. Range: [0.0, 1.0]. Default: 0.5

    Returns:
        List[Dict[str, float]]: List of dictionaries (one per batch item) with metrics:
            - mean_iou: Mean IoU of correctly matched objects
            - precision: Fraction of predicted objects that match ground truth
            - recall: Fraction of ground truth objects that were detected
            - f1_score: Harmonic mean of precision and recall

    """
    true_mask = standardize_mask(true_mask)
    pred_mask = standardize_mask(pred_mask)

    batch_metrics = []
    for b in range(true_mask.shape[0]):
        true_slice = true_mask[b, 0]
        pred_slice = pred_mask[b, 0]

        # Get connected components
        true_labeled, true_n = label(true_slice > 0)
        pred_labeled, pred_n = label(pred_slice > 0)

        # Handle special cases
        if true_n == 0 and pred_n == 0:
            batch_metrics.append({
                "mean_iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0
            })
            continue

        if true_n == 0:
            batch_metrics.append({
                "mean_iou": 0.0,
                "precision": 0.0,
                "recall": 1.0,
                "f1_score": 0.0
            })
            continue

        if pred_n == 0:
            batch_metrics.append({
                "mean_iou": 0.0,
                "precision": 1.0,
                "recall": 0.0,
                "f1_score": 0.0
            })
            continue

        # Compute IoU matrix
        iou_matrix = np.zeros((true_n, pred_n))
        for i in range(1, true_n + 1):
            true_obj = true_labeled == i
            for j in range(1, pred_n + 1):
                pred_obj = pred_labeled == j
                intersection = np.logical_and(true_obj, pred_obj).sum()
                union = np.logical_or(true_obj, pred_obj).sum()
                iou_matrix[i - 1, j - 1] = intersection / union if union > 0 else 0

        # Find optimal matching
        true_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        matched_ious = iou_matrix[true_indices, pred_indices]
        valid_matches = matched_ious >= iou_threshold
        true_positives = np.sum(valid_matches)

        # Calculate metrics
        precision = true_positives / pred_n if pred_n > 0 else 0
        recall = true_positives / true_n if true_n > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        mean_iou = matched_ious[valid_matches].mean() if true_positives > 0 else 0.0

        batch_metrics.append({
            "mean_iou": float(mean_iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        })

    return batch_metrics


class ChannelSpec:
    """Specification for channel selection in cell segmentation models.

    This class defines which imaging channels should be used for segmentation,
    specifically mapping the nuclear stain and membrane/cytoplasm markers required
    by segmentation algorithms like Mesmer.

    The channel names provided must exactly match names in the ImageData object's
    channel_names attribute. This ensures correct channel selection even when
    the raw data contains additional channels not used for segmentation.

    Attributes:
        nuclear (str): Name of the nuclear staining channel (e.g., "DAPI", "Hoechst")
        membrane (List[str]): List of channels to combine for membrane/cytoplasm signal.
                            Multiple channels will be summed to create a composite
                            membrane signal.

    Examples:
        >>> # Basic usage with single membrane channel
        >>> spec = ChannelSpec(nuclear="DAPI", membrane=["CD44"])
        
        >>> # Using multiple membrane markers
        >>> spec = ChannelSpec(
        ...     nuclear="Hoechst",
        ...     membrane=["Na/K-ATPase", "E-cadherin"]
        ... )
    """
    def __init__(self, nuclear: str, membrane: List[str]):
        self.nuclear = nuclear
        self.membrane = membrane


class BaseSegmenter(ABC):
    """Abstract base class defining the interface for cell segmentation models.

    This class provides a standardized interface that all segmentation model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction while remaining model-agnostic.

    The interface is designed to support both:
    1. Channel-based segmentation (e.g., Mesmer) that requires specific imaging
       channels like nuclear and membrane markers
    2. Direct image segmentation (e.g., SAM2) that can work with any image input

    All implementations must handle batched inputs and outputs consistently,
    following the shape conventions defined in the ImageData class.

    Methods:
        preprocess: Prepare raw image data for model input
        predict: Generate segmentation predictions for the input images

    Example Implementation:
        >>> class MySegmenter(BaseSegmenter):
        ...     def preprocess(self, image_data, channel_spec=None):
        ...         # Implementation
        ...         pass
        ...
        ...     def predict(self, image_data, channel_spec=None, **kwargs):
        ...         # Implementation
        ...         pass

    Notes:
        - All methods must preserve batch dimensions
        - Implementations should handle both single-channel and multi-channel inputs
        - Error handling should be comprehensive and informative
        - Memory efficiency should be considered for large batches
    """
    @abstractmethod
    def preprocess(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare images for model input.
        
        Returns preprocessed numpy array regardless of underlying framework.
        
        Args:
            image_data: Input images and metadata
            channel_spec: Optional channel specification
            
        Returns:
            Preprocessed array in model's expected format
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs
    ) -> ImageData:
        """Generate segmentation predictions.
        
        Takes framework-agnostic input and returns framework-agnostic output.
        
        Args:
            image_data: Input images and metadata
            channel_spec: Optional channel specification
            **kwargs: Additional model parameters
            
        Returns:
            ImageData with predictions populated
        """
        pass


class MesmerSegmenter(BaseSegmenter):
    """Cell segmentation implementation using the Mesmer deep learning model.

    This class implements whole-cell segmentation using the Mesmer model, which uses
    both nuclear and membrane/cytoplasm channels to accurately identify cell boundaries.
    The implementation handles all necessary preprocessing and ensures proper formatting
    of inputs/outputs while supporting batched processing.

    Key Features:
        - Automatic channel preprocessing and normalization
        - Support for multiple membrane markers
        - Built-in error handling for missing channels
        - Batched processing for efficient throughput
        - Configurable model parameters

    Attributes:
        model (Mesmer): Instance of the Mesmer model

    Example:
        >>> # Initialize segmenter
        >>> segmenter = MesmerSegmenter(model_kwargs={'image_size': 256})
        
        >>> # For multichannel data with specific channels
        >>> channel_spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
        >>> segmented_data = segmenter.predict(
        ...     image_data=image_data,
        ...     channel_spec=channel_spec
        ... )
        
        >>> # For single-channel data
        >>> segmented_data = segmenter.predict(image_data)

    Notes:
        - The model expects normalized input images
        - Nuclear and membrane channels are automatically balanced
        - Batch processing preserves memory by processing sequentially
        - Error messages are detailed for debugging purposes
    """
    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize Mesmer model.
        
        Args:
            model_kwargs: Optional model configuration
        """
        self.model = Mesmer(**(model_kwargs or {}))

    def _to_tensorflow(self, array: np.ndarray) -> tf.Tensor:
        """Convert numpy array to TensorFlow tensor."""
        return tf.convert_to_tensor(array)

    def _from_tensorflow(self, tensor: tf.Tensor) -> np.ndarray:
        """Convert TensorFlow tensor to numpy array."""
        return tensor.numpy()

    def preprocess(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare images for Mesmer input."""
        raw = image_data.raw  # (B, C, H, W)
        batch_size = raw.shape[0]
        processed_batch = []

        for b in range(batch_size):
            single_image = raw[b]  # (C, H, W)

            # Handle single/multi-channel
            if single_image.shape[0] == 1:
                processed = np.stack([single_image[0]] * 2, axis=0)
            else:
                if channel_spec is None:
                    raise ValueError("Channel specification required for multi-channel input")
                
                # Get nuclear and membrane channels
                try:
                    nuclear_idx = image_data.channel_names.index(channel_spec.nuclear)
                    membrane_indices = [
                        image_data.channel_names.index(ch) for ch in channel_spec.membrane
                    ]
                except ValueError as e:
                    raise ValueError(f"Channel not found: {str(e)}")
                
                nuclear = single_image[nuclear_idx]
                membrane = np.sum([single_image[idx] for idx in membrane_indices], axis=0)
                processed = np.stack([nuclear, membrane], axis=0)

            processed_batch.append(processed)

        # Stack and transpose to (B, H, W, C)
        return np.stack(processed_batch, axis=0).transpose(0, 2, 3, 1)

    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs
    ) -> ImageData:
        """Generate Mesmer segmentation predictions."""
        # Preprocess to numpy
        processed_img = self.preprocess(image_data, channel_spec)
        
        # Convert to TensorFlow
        tf_input = self._to_tensorflow(processed_img)
        
        # Run prediction
        try:
            labels = self.model.predict(
                tf_input,
                compartment="nuclear",
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Mesmer segmentation failed: {str(e)}") from e

        # Standardize output
        labels = standardize_mask(labels)

        return replace(image_data, predicted_masks=labels)


class SAM2Segmenter(BaseSegmenter):
    """Cell segmentation implementation using the SAM2 model.

    This class provides whole-cell segmentation capabilities using SAM2's automatic
    mask generation. It adapts SAM2's output format to match the ImageData structure
    used in the rest of the codebase while supporting efficient batch processing.

    The implementation automatically handles various input formats and provides
    extensive configuration options for the mask generation process. It includes
    built-in filtering of large masks that likely represent background regions.

    Attributes:
        model (SAM2Base): The SAM2 model instance
        mask_generator (SAM2AutomaticMaskGenerator): The automatic mask generator
        device (str): Device to run the model on ('cuda' or 'cpu')

    Args:
        model_cfg (str): Path to SAM2 model configuration file
        checkpoint_path (str): Path to SAM2 model weights
        points_per_side (int, optional): Number of points to sample along each side.
            Defaults to 32.
        points_per_batch (int): Number of points to process in parallel.
            Defaults to 64.
        pred_iou_thresh (float): Threshold for predicted mask quality.
            Range: [0.0, 1.0]. Defaults to 0.8.
        stability_score_thresh (float): Threshold for mask stability.
            Range: [0.0, 1.0]. Defaults to 0.95.
        stability_score_offset (float): Offset for stability score calculation.
            Defaults to 1.0.
        box_nms_thresh (float): IoU threshold for box NMS.
            Range: [0.0, 1.0]. Defaults to 0.7.
        crop_n_layers (int): Number of crop layers to use. Defaults to 0.
        crop_n_points_downscale_factor (int): Factor to reduce points in deeper crops.
            Defaults to 1.
        min_mask_region_area (int): Minimum area for mask regions. Defaults to 0.
        use_m2m (bool): Whether to use mask-to-mask refinement. Defaults to False.
        device (str): Device to run model on ('cuda' or 'cpu').
            Defaults to 'cuda' if available.

    Example:
        >>> segmenter = SAM2Segmenter(
        ...     model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        ...     checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        ...     points_per_side=64,
        ...     pred_iou_thresh=0.9
        ... )
        >>> segmented_data = segmenter.predict(image_data)

    Notes:
        - Automatically converts various input formats to RGB
        - Filters out masks larger than 5000 pixels as potential background
        - Processes images batch-wise for memory efficiency
        - Provides detailed logging of the segmentation process
        - Supports both CPU and GPU execution
    """
    def __init__(
        self,
        model_cfg: str,
        checkpoint_path: str,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        use_m2m: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize SAM2 model and mask generator."""
        self.device = device
        self.model = build_sam2(
            model_cfg,
            checkpoint_path,
            device=device,
            apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            use_m2m=use_m2m,
            output_mode="binary_mask",
        )

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.from_numpy(array).to(self.device)

    def _from_torch(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.cpu().numpy()

    def preprocess(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare images for SAM2 input."""
        raw = image_data.raw  # (B, C, H, W)
        batch_size = raw.shape[0]
        processed_batch = []

        for b in range(batch_size):
            single_image = raw[b]  # (C, H, W)

            # Convert to RGB format
            if single_image.shape[0] == 1:
                processed = np.stack([single_image[0]] * 3, axis=0)
            elif single_image.shape[0] == 3:
                processed = single_image
            else:
                processed = np.zeros((3, *single_image.shape[1:]))
                for i in range(min(single_image.shape[0], 3)):
                    processed[i] = single_image[i]

            # Normalize to [0, 255]
            processed = processed.astype(np.float32)
            if processed.max() > 0:
                processed = (
                    (processed - processed.min())
                    / (processed.max() - processed.min())
                    * 255
                )
            processed = processed.astype(np.uint8)
            
            # Convert to (H, W, 3)
            processed = np.transpose(processed, (1, 2, 0))
            processed_batch.append(processed)

        return np.stack(processed_batch, axis=0)

    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs
    ) -> ImageData:
        """Generate SAM2 segmentation predictions."""
        processed_imgs = self.preprocess(image_data, channel_spec)
        batch_size = processed_imgs.shape[0]
        batch_masks = []

        for b in range(batch_size):
            try:
                masks = self.mask_generator.generate(processed_imgs[b])
            except Exception as e:
                raise RuntimeError(f"SAM2 segmentation failed for batch {b}: {str(e)}")

            if not masks:
                combined_mask = np.zeros(processed_imgs[b].shape[:2], dtype=np.int32)
            else:
                combined_mask = np.zeros(processed_imgs[b].shape[:2], dtype=np.int32)
                idx = 1
                for mask_data in masks:
                    mask = mask_data["segmentation"]
                    area = mask_data["area"]
                    if area > 5000:  # Skip potential background
                        logging.info(f"Batch {b}: Skipping large mask")
                        continue
                    combined_mask[mask] = idx
                    idx += 1

            batch_masks.append(combined_mask)

        # Stack and standardize
        combined_masks = np.stack(batch_masks, axis=0)
        combined_masks = standardize_mask(combined_masks)

        return replace(image_data, predicted_masks=combined_masks)