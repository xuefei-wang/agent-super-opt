from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Sequence
from dataclasses import replace
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment
import pandas as pd
from pathlib import Path
import torch

from deepcell.applications import Mesmer
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


from .data_io import ImageData, standardize_mask


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


def calculate_object_metrics(
    true_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate instance segmentation metrics between two masks.

    Args:
        true_mask: Ground truth segmentation mask (1, H, W)
        pred_mask: Predicted segmentation mask (1, H, W)
        iou_threshold: IoU threshold for considering an object as correctly detected

    Returns:
        Dictionary containing computed metrics
    """
    # Ensure masks are standardized
    true_mask = standardize_mask(true_mask)
    pred_mask = standardize_mask(pred_mask)

    # Use first channel for processing
    true_mask = true_mask[0]  # Convert (1, H, W) to (H, W)
    pred_mask = pred_mask[0]  # Convert (1, H, W) to (H, W)

    # Get unique objects by considering connected components
    from scipy.ndimage import label

    # Label connected components in each mask
    true_labeled, true_n = label(true_mask > 0)
    pred_labeled, pred_n = label(pred_mask > 0)

    # Handle special cases
    if true_n == 0 and pred_n == 0:
        return {"mean_iou": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0}

    if true_n == 0:
        return {"mean_iou": 0.0, "precision": 0.0, "recall": 1.0, "f1_score": 0.0}

    if pred_n == 0:
        return {"mean_iou": 0.0, "precision": 1.0, "recall": 0.0, "f1_score": 0.0}

    # Compute IoU matrix
    iou_matrix = np.zeros((true_n, pred_n))
    for i in range(1, true_n + 1):
        true_obj = true_labeled == i
        for j in range(1, pred_n + 1):
            pred_obj = pred_labeled == j
            intersection = np.logical_and(true_obj, pred_obj).sum()
            union = np.logical_or(true_obj, pred_obj).sum()
            iou_matrix[i - 1, j - 1] = intersection / union if union > 0 else 0

    # Find optimal matching using Hungarian algorithm
    true_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    # Get IoUs for matched pairs
    matched_ious = iou_matrix[true_indices, pred_indices]

    # Only count matches that exceed the IoU threshold
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

    # Calculate mean IoU only for valid matches
    mean_iou = matched_ious[valid_matches].mean() if true_positives > 0 else 0.0

    return {
        "mean_iou": float(mean_iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


class BaseSegmenter(ABC):
    """
    Abstract base class defining the interface for cell segmentation models.

    This class provides a standardized interface that all segmentation model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction.

    The interface is designed to be model-agnostic while ensuring consistent
    handling of input data and segmentation results across different implementations.
    It supports both channel-based segmentation (e.g., Mesmer) and direct image
    segmentation (e.g., SAM2).

    Example:
        class MySegmenter(BaseSegmenter):
            def preprocess(self, image_data, channel_spec=None):
                # Implementation
                pass

            def predict(self, image_data, channel_spec=None, **kwargs):
                # Implementation
                pass
    """

    @abstractmethod
    def preprocess(
        self, image_data: ImageData, channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare image data for model input.

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Optional specification of required channels. If None,
                        uses all available channels or default processing.

        Returns:
            np.ndarray: Processed image ready for model input

        Raises:
            ValueError: If data format is invalid or specified channels not found
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs,
    ) -> ImageData:
        """Perform segmentation prediction.

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Optional specification of required channels
            **kwargs: Additional model-specific parameters

        Returns:
            ImageData: Updated copy of input with predicted_mask field populated

        Raises:
            RuntimeError: If segmentation fails
            ValueError: If input data format is invalid
        """
        pass

    @staticmethod
    def calculate_metrics(
        images: Sequence[ImageData], iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate segmentation quality metrics across multiple images."""
        metrics_list = []

        for img in images:
            if img.mask is None:
                raise ValueError("Ground truth mask missing")
            if img.predicted_mask is None:
                raise ValueError("Predicted mask missing")

            # Masks are already standardized to (1, H, W) in ImageData
            metrics = calculate_object_metrics(
                img.mask, img.predicted_mask, iou_threshold=iou_threshold
            )
            metrics_list.append(metrics)

        # Average metrics across all images
        df = pd.DataFrame(metrics_list)
        return df.mean().to_dict()


class MesmerSegmenter(BaseSegmenter):
    """Cell segmentation implementation using the Mesmer deep learning model.

    This class provides whole-cell segmentation capabilities using the Mesmer model,
    which uses both nuclear and membrane/cytoplasm channels to accurately identify
    cell boundaries. The implementation handles all necessary preprocessing and
    ensures proper formatting of inputs/outputs.

    Key Features:
    - Automatic channel preprocessing and normalization
    - Support for multiple membrane markers
    - Support for single-channel inputs
    - Built-in error handling for missing channels
    - Configurable model parameters

    Example:
        >>> # For multichannel data with specific channels
        >>> channel_spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
        >>> segmenter = MesmerSegmenter()
        >>> segmented_data = segmenter.predict(
        ...     image_data=image_data,
        ...     channel_spec=channel_spec
        ... )

        >>> # For single-channel data
        >>> segmenter = MesmerSegmenter()
        >>> segmented_data = segmenter.predict(image_data)
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

    def preprocess(
        self, image_data: ImageData, channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare channels for Mesmer input.

        This method handles various input formats:
        1. Single-channel input: Uses the same channel for both nuclear and membrane signals
        2. Multichannel with channel_spec: Uses specified channels
        3. Multichannel without channel_spec: Attempts to automatically select appropriate channels

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Optional specification of nuclear and membrane channels

        Returns:
            np.ndarray: Processed image with shape (2, height, width) containing
                       [nuclear_channel, membrane_channel]

        Raises:
            ValueError: If required channels cannot be found or selected
        """
        raw = image_data.raw

        # Handle single-channel input
        if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[-1] == 1):
            if raw.ndim == 3:
                raw = raw[..., 0]  # Convert (H, W, 1) to (H, W)

            # Use the same channel for both nuclear and membrane signals
            nuclear_img = raw
            membrane_img = raw

        # Handle multichannel input
        elif raw.ndim == 3 and raw.shape[0] > 1:  # (C, H, W) format
            if channel_spec is None:
                # Attempt to automatically select channels
                nuclear_patterns = ["DAPI", "Hoechst", "H3342", "nuclear"]
                membrane_patterns = ["membrane", "CD44", "Na/K", "WGA"]

                if image_data.channel_names is None:
                    # If no channel names, use first channel for nuclear and sum of others for membrane
                    nuclear_idx = 0
                    membrane_indices = list(range(1, raw.shape[0]))
                else:
                    # Find nuclear channel
                    nuclear_idx = -1
                    for pattern in nuclear_patterns:
                        matching = [
                            i
                            for i, name in enumerate(image_data.channel_names)
                            if pattern.lower() in name.lower()
                        ]
                        if matching:
                            nuclear_idx = matching[0]
                            break

                    if nuclear_idx == -1:
                        nuclear_idx = 0  # Default to first channel if no match

                    # Find membrane channels
                    membrane_indices = []
                    for pattern in membrane_patterns:
                        matching = [
                            i
                            for i, name in enumerate(image_data.channel_names)
                            if pattern.lower() in name.lower()
                        ]
                        membrane_indices.extend(matching)

                    if not membrane_indices:
                        membrane_indices = list(
                            range(1, raw.shape[0])
                        )  # Use all non-nuclear channels
            else:
                # Use specified channels
                try:
                    nuclear_idx = image_data.channel_names.index(channel_spec.nuclear)
                    membrane_indices = [
                        image_data.channel_names.index(ch)
                        for ch in channel_spec.membrane
                    ]
                except ValueError as e:
                    raise ValueError(f"Channel not found: {str(e)}")

            # Extract channels
            nuclear_img = raw[nuclear_idx]
            membrane_img = np.zeros_like(nuclear_img)
            for idx in membrane_indices:
                membrane_img += raw[idx]

        else:
            raise ValueError(f"Unsupported input shape: {raw.shape}")

        # Normalize both channels
        nuclear_img = self._normalize(nuclear_img)
        membrane_img = self._normalize(membrane_img)

        return np.stack([nuclear_img, membrane_img])

    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs,
    ) -> ImageData:
        """Segment cells using Mesmer."""
        # Preprocess channels
        processed_img = self.preprocess(image_data, channel_spec)

        # Convert to Mesmer's expected format (batch, height, width, channels)
        model_input = np.moveaxis(processed_img, 0, -1)
        model_input = model_input[np.newaxis, ...]  # Add batch dimension

        # Run prediction
        try:
            labels = self.model.predict(model_input, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Mesmer segmentation failed: {str(e)}") from e

        # Standardize output to (1, H, W) format
        labels = standardize_mask(labels)

        return replace(image_data, predicted_mask=labels)

    @staticmethod
    def _normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize image to [0, 1] range with handling for edge cases.

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


class SAM2Segmenter(BaseSegmenter):
    """
    Cell segmentation implementation using the SAM2 model.

    This class provides whole-cell segmentation capabilities using SAM2's automatic
    mask generation. It adapts SAM2's output format to match the ImageData structure
    used in the rest of the codebase.

    Attributes:
        model (SAM2Base): The SAM2 model instance
        mask_generator (SAM2AutomaticMaskGenerator): The automatic mask generator
        device (str): Device to run the model on ('cuda' or 'cpu')

    Example:
        >>> segmenter = SAM2Segmenter(
        ...     model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        ...     checkpoint_path="../checkpoints/sam2.1_hiera_large.pt",
        ...     points_per_side=64
        ... )
        >>> segmented_data = segmenter.predict(
        ...     image_data=image_data,
        ...     channel_spec=channel_spec
        ... )
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        points_per_side: int = 64,
        points_per_batch: int = 128,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.92,
        stability_score_offset: float = 0.7,
        crop_n_layers: int = 1,
        box_nms_thresh: float = 0.7,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: float = 25.0,
        use_m2m: bool = True,
    ):
        """Initialize SAM2 model and mask generator.

        Args:
            model_cfg: Path to SAM2 model configuration file
            checkpoint_path: Path to SAM2 model weights
            device: Device to run model on ('cuda' or 'cpu')
            points_per_side: Number of points to sample along each side
            points_per_batch: Number of points to process in parallel
            pred_iou_thresh: Threshold for predicted mask quality
            stability_score_thresh: Threshold for mask stability
            stability_score_offset: Offset for stability score calculation
            crop_n_layers: Number of crop layers to use
            box_nms_thresh: IoU threshold for box NMS
            crop_n_points_downscale_factor: Factor to reduce points in deeper crops
            min_mask_region_area: Minimum area for mask regions
            use_m2m: Whether to use mask-to-mask refinement

        Raises:
            RuntimeError: If model initialization fails
        """
        self.device = device

        # Initialize SAM2 model
        try:
            self.model = build_sam2(
                model_cfg, checkpoint_path, device=device, apply_postprocessing=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM2 model: {str(e)}")

        # Initialize mask generator
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

    def preprocess(
        self, image_data: ImageData, channel_spec: Optional[ChannelSpec] = None
    ) -> np.ndarray:
        """Prepare image data for SAM2 input.

        For SAM2, we need RGB-format input. The method will:
        1. For multichannel data: Take first 3 channels or pad with zeros
        2. For single-channel data: Replicate to 3 channels
        3. For RGB data: Use as is

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Not used by SAM2, included for interface consistency

        Returns:
            np.ndarray: Processed image with shape (height, width, 3)
        """
        raw = image_data.raw

        # Handle different input formats
        if raw.ndim == 2:  # Single channel
            processed = np.stack([raw] * 3, axis=-1)
        elif raw.ndim == 3 and raw.shape[-1] == 1:  # Single channel with extra dim
            processed = np.concatenate([raw] * 3, axis=-1)
        elif raw.ndim == 3 and raw.shape[-1] == 3:  # Already RGB
            processed = raw
        else:  # Multichannel data
            processed = np.zeros((*raw.shape[1:], 3))
            for i in range(min(raw.shape[0], 3)):
                processed[..., i] = raw[i]

        # Normalize to [0, 255]
        processed = processed.astype(np.float32)
        if processed.max() > 0:
            processed = (
                (processed - processed.min())
                / (processed.max() - processed.min())
                * 255
            )

        return processed.astype(np.uint8)

    def predict(
        self,
        image_data: ImageData,
        channel_spec: Optional[ChannelSpec] = None,
        **kwargs,
    ) -> ImageData:
        """Perform segmentation using SAM2."""
        # Preprocess image
        processed_img = self.preprocess(image_data)

        # Generate masks
        try:
            masks = self.mask_generator.generate(processed_img)
        except Exception as e:
            raise RuntimeError(f"SAM2 segmentation failed: {str(e)}")

        if not masks:
            # No masks found - return empty mask with correct shape
            empty_mask = np.zeros((1, *processed_img.shape[:2]), dtype=np.int32)
            return replace(image_data, predicted_mask=empty_mask)

        # Convert SAM2 output to our format
        combined_mask = np.zeros(processed_img.shape[:2], dtype=np.int32)
        for idx, mask_data in enumerate(masks, start=1):
            mask = mask_data["segmentation"]
            combined_mask[mask] = idx

        # Standardize to (1, H, W) format
        combined_mask = standardize_mask(combined_mask)

        return replace(image_data, predicted_mask=combined_mask)
