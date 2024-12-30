from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Sequence
from dataclasses import replace
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment
import pandas as pd

from deepcell.applications import Mesmer

from .data_io import ImageData


class ChannelSpec:
    """
    Specification for channel selection in cell segmentation models.

    This class defines which imaging channels should be used for segmentation,
    specifically mapping the nuclear stain and membrane/cytoplasm markers required
    by segmentation algorithms like Mesmer. Each channel name must exactly match
    a channel name in the ImageData object.

    Attributes:
        nuclear (str): Name of the nuclear staining channel (e.g., "DAPI", "Hoechst")
        membrane (List[str]): List of channels to combine for membrane/cytoplasm signal.
                            Multiple channels will be summed to create a composite signal.

    Example:
        >>> spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
    """

    def __init__(self, nuclear: str, membrane: List[str]):
        """Initialize channel specification.

        Args:
            nuclear: Name of nuclear channel (must match a name in ImageData.channel_names)
            membrane: List of channel names to combine for membrane signal
                     (each must match a name in ImageData.channel_names)
        """
        self.nuclear = nuclear
        self.membrane = membrane


def calculate_object_metrics(true_mask: np.ndarray, pred_mask: np.ndarray, 
                           iou_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate instance segmentation metrics between two masks.
    
    This function computes key metrics for instance segmentation:
    - Mean IoU: Average Intersection over Union for matched objects
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: Harmonic mean of precision and recall
    
    Args:
        true_mask: Ground truth segmentation mask
        pred_mask: Predicted segmentation mask
        iou_threshold: IoU threshold for considering an object as correctly detected
        
    Returns:
        Dictionary containing computed metrics
    """
    # Get unique objects (excluding background=0)
    true_objects = np.unique(true_mask)[1:]
    pred_objects = np.unique(pred_mask)[1:]
    
    if len(true_objects) == 0 and len(pred_objects) == 0:
        return {
            'mean_iou': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0
        }
    
    if len(true_objects) == 0:
        return {
            'mean_iou': 0.0,
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0
        }
    
    if len(pred_objects) == 0:
        return {
            'mean_iou': 0.0,
            'precision': 1.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(true_objects), len(pred_objects)))
    for i, true_id in enumerate(true_objects):
        true_obj = (true_mask == true_id)
        for j, pred_id in enumerate(pred_objects):
            pred_obj = (pred_mask == pred_id)
            intersection = np.logical_and(true_obj, pred_obj).sum()
            union = np.logical_or(true_obj, pred_obj).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Find optimal matching using Hungarian algorithm
    true_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    
    # Calculate metrics
    matches = iou_matrix[true_indices, pred_indices] >= iou_threshold
    true_positives = np.sum(matches)
    
    precision = true_positives / len(pred_objects) if len(pred_objects) > 0 else 0
    recall = true_positives / len(true_objects) if len(true_objects) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean IoU for matched objects
    matched_ious = iou_matrix[true_indices[matches], pred_indices[matches]]
    mean_iou = matched_ious.mean() if len(matched_ious) > 0 else 0
    
    return {
        'mean_iou': float(mean_iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }


class BaseSegmenter(ABC):
    """
    Abstract base class defining the interface for cell segmentation models.

    This class provides a standardized interface that all segmentation model
    implementations must follow. It defines the core methods needed for channel
    preprocessing and cell segmentation prediction.

    The interface is designed to be model-agnostic while ensuring consistent
    handling of input data and segmentation results across different implementations.

    Example:
        class MySegmenter(BaseSegmenter):
            def preprocess_channels(self, image_data, channel_spec):
                # Implementation
                pass

            def predict(self, image_data, channel_spec, **kwargs):
                # Implementation
                pass
    """

    @abstractmethod
    def preprocess_channels(
        self, image_data: ImageData, channel_spec: ChannelSpec
    ) -> np.ndarray:
        """Prepare channels according to model requirements.

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Specification of required channels

        Returns:
            np.ndarray: Processed image ready for model input

        Raises:
            ValueError: If specified channels are not found in image_data.channel_names
        """
        pass

    @abstractmethod
    def predict(self, image_data: ImageData, channel_spec: ChannelSpec, **kwargs) -> ImageData:
        pass
    
    @staticmethod
    def calculate_metrics(images: Sequence[ImageData], 
                         iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate segmentation quality metrics across multiple images.
        
        This method computes average segmentation metrics by comparing predicted masks
        against ground truth masks. It requires both ground truth masks and
        predicted masks to be present in the ImageData objects.
        
        Args:
            images: List of ImageData objects containing both ground truth and
                   predicted masks
            iou_threshold: IoU threshold for considering an object as correctly detected
            
        Returns:
            Dictionary containing averaged metrics across all images:
            - mean_iou: Mean Intersection over Union
            - precision: True Positives / (True Positives + False Positives)
            - recall: True Positives / (True Positives + False Negatives)
            - f1_score: Harmonic mean of precision and recall
            
        Raises:
            ValueError: If masks are missing or invalid
        """
        metrics_list = []
        
        for img in images:
            if img.mask is None:
                raise ValueError("Ground truth mask missing")
            if img.predicted_mask is None:
                raise ValueError("Predicted mask missing")
                
            # Ensure masks are 2D
            true_mask = img.mask[0] if img.mask.ndim == 3 else img.mask
            pred_mask = img.predicted_mask[0] if img.predicted_mask.ndim == 3 else img.predicted_mask
            
            # Calculate metrics for this image
            img_metrics = calculate_object_metrics(
                true_mask, 
                pred_mask,
                iou_threshold=iou_threshold
            )
            metrics_list.append(img_metrics)
        
        # Average metrics across all images
        df = pd.DataFrame(metrics_list)
        return df.mean().to_dict()


class MesmerSegmenter(BaseSegmenter):
    """
    Cell segmentation implementation using the Mesmer deep learning model.

    This class provides whole-cell segmentation capabilities using the Mesmer model,
    which uses both nuclear and membrane/cytoplasm channels to accurately identify
    cell boundaries. The implementation handles all necessary preprocessing and
    ensures proper formatting of inputs/outputs.

    Key Features:
    - Automatic channel preprocessing and normalization
    - Support for multiple membrane markers
    - Built-in error handling for missing channels
    - Configurable model parameters

    Example:
        >>> # Define channels to use
        >>> channel_spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
        >>>
        >>> # Initialize and run segmentation
        >>> segmenter = MesmerSegmenter()
        >>> segmented_data = segmenter.predict(
        ...     image_data=image_data,
        ...     channel_spec=channel_spec
        ... )
    """

    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize Mesmer model.

        Args:
            model_kwargs: Optional dictionary of arguments passed to Mesmer initialization.
                        See Mesmer documentation for available parameters.

        Raises:
            RuntimeError: If Mesmer model initialization fails
        """
        self.model = Mesmer(**(model_kwargs or {}))

    def preprocess_channels(
        self, image_data: ImageData, channel_spec: ChannelSpec
    ) -> np.ndarray:
        """Combine appropriate channels for Mesmer input.

        This method extracts the nuclear channel and combines specified membrane
        channels into a single membrane signal. Both channels are normalized
        independently to the range [0, 1].

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Specification of required channels

        Returns:
            np.ndarray: Processed image with shape (2, height, width) containing
                       [nuclear_channel, combined_membrane_channels]

        Raises:
            ValueError: If any specified channel is not found in image_data.channel_names
        """
        # Get nuclear channel
        try:
            nuc_idx = image_data.channel_names.index(channel_spec.nuclear)
            nuclear_img = image_data.raw[nuc_idx]
        except ValueError:
            raise ValueError(
                f"Nuclear channel '{channel_spec.nuclear}' not found in provided channels"
            )

        # Combine membrane channels
        membrane_indices = []
        for channel in channel_spec.membrane:
            try:
                idx = image_data.channel_names.index(channel)
                membrane_indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Membrane channel '{channel}' not found in provided channels"
                )

        membrane_img = np.zeros_like(nuclear_img)
        for idx in membrane_indices:
            membrane_img += image_data.raw[idx]

        # Normalize each channel independently
        nuclear_img = self._normalize(nuclear_img)
        membrane_img = self._normalize(membrane_img)

        return np.stack([nuclear_img, membrane_img])

    def predict(
        self, image_data: ImageData, channel_spec: ChannelSpec, **kwargs
    ) -> ImageData:
        """Segment cells using Mesmer.

        This method performs whole-cell segmentation using the Mesmer model.
        It automatically preprocesses the channels and returns an updated
        ImageData object with the segmentation results.

        Args:
            image_data: ImageData object containing raw image and metadata.
                       The raw image should have shape (channels, height, width)
            channel_spec: Specification of nuclear and membrane channels to use
            **kwargs: Additional arguments passed to Mesmer's predict method.
                     See Mesmer documentation for available parameters.

        Returns:
            ImageData: Updated copy of input ImageData with predicted_mask field
                      populated. The mask has shape (1, height, width) where each
                      integer represents a unique cell.

        Raises:
            ValueError: If required channels are not found
            RuntimeError: If segmentation fails
        """
        # Preprocess channels
        processed_img = self.preprocess_channels(image_data, channel_spec)

        # Convert to Mesmer's expected format (batch, height, width, channels)
        model_input = np.moveaxis(processed_img, 0, -1)
        model_input = model_input[np.newaxis, ...]  # Add batch dimension

        # Run prediction
        try:
            labels = self.model.predict(model_input, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Mesmer segmentation failed: {str(e)}") from e

        # Create new ImageData with predicted mask
        return replace(
            image_data,
            predicted_mask=np.squeeze(labels),
        )

    @staticmethod
    def _normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize image to [0, 1] range with handling for edge cases.

        This is an internal helper method used during channel preprocessing.

        Args:
            img (np.ndarray): Input image to normalize
            eps (float): Small epsilon value to prevent division by zero

        Returns:
            np.ndarray: Normalized image in [0, 1] range
        """
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()

        if abs(img_max - img_min) < eps:
            return np.zeros_like(img)

        return (img - img_min) / (img_max - img_min)
