import zarr
import numpy as np
from typing import List, Dict, Optional, Union, Sequence, Set, Tuple
from dataclasses import dataclass
import logging
import torch
from pathlib import Path


import numpy as np
from typing import Tuple, Optional


def standardize_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Standardize mask shape to (1, height, width) format.

    Args:
        mask: Input mask with shape (height, width), (height, width, 1),
              (1, height, width), (1, height, width, 1) or similar variations

    Returns:
        np.ndarray: Standardized mask with shape (1, height, width)

    Raises:
        ValueError: If mask dimensions are invalid
    """
    if mask is None:
        return None

    # Handle 2D case
    if mask.ndim == 2:
        return mask[np.newaxis, ...]

    # Handle 3D case
    elif mask.ndim == 3:
        if mask.shape[0] == 1:  # Already (1, H, W)
            return mask
        elif mask.shape[-1] == 1:  # (H, W, 1)
            return np.transpose(mask, (2, 0, 1))
        elif mask.shape[0] > 1:  # Multiple channels - take first channel
            return mask[0:1, ...]
        else:
            raise ValueError(
                f"Invalid mask shape {mask.shape}. Cannot standardize to (1, H, W) format."
            )
    elif mask.ndim == 4:
        if mask.shape[0] == 1 and mask.shape[-1] == 1:
            return mask[..., 0]
        else:
            raise ValueError(
                f"Invalid mask shape {mask.shape}. Cannot standardize to (1, H, W) format."
            )
    else:
        raise ValueError(
            f"Invalid mask dimensionality {mask.ndim}. Expected 2 or 3 dimensions."
        )


def standardize_raw_image(
    raw: np.ndarray, is_multichannel: bool = False
) -> Tuple[np.ndarray, bool]:
    """Standardize raw image shape to (C, height, width) or (1, height, width) format.

    Args:
        raw: Input image array
        is_multichannel: Whether the image should be treated as multichannel

    Returns:
        Tuple[np.ndarray, bool]: (Standardized array, is_multichannel flag)

    Raises:
        ValueError: If dimensions are invalid
    """
    if raw is None:
        return None, is_multichannel

    # Handle 2D case (single channel)
    if raw.ndim == 2:
        return raw[np.newaxis, ...], False

    # Handle 3D case
    elif raw.ndim == 3:
        if raw.shape[-1] == 1:  # (H, W, 1) -> (1, H, W)
            return np.transpose(raw, (2, 0, 1)), False
        elif raw.shape[0] == 1:  # Already (1, H, W)
            return raw, False
        else:  # Multichannel: ensure (C, H, W)
            if raw.shape[-1] > 1 and raw.shape[0] != min(raw.shape):  # (H, W, C)
                raw = np.transpose(raw, (2, 0, 1))
            return raw, True

    else:
        raise ValueError(
            f"Invalid raw image dimensionality {raw.ndim}. Expected 2 or 3 dimensions."
        )


@dataclass
class ImageData:
    """Container for biological image data and associated metadata.
    
    This class provides a standardized structure for storing and managing biological
    image data along with related annotations and predictions. It handles data
    validation and format standardization automatically.

    The class standardizes array shapes as follows:
    - Raw images: (C, H, W) format where C is number of channels
    - Masks: (1, H, W) format for both ground truth and predictions

    Attributes:
        raw (np.ndarray): Raw image data in (C, H, W) format. For multichannel data,
            C corresponds to different imaging channels (e.g., fluorescence markers).
            For single-channel data, C=1.
        
        image_id (Union[int, str]): Unique identifier for the image. Can be numeric
            index or string identifier.
        
        channel_names (Optional[List[str]]): Names of imaging channels in order matching
            raw data channels. For multichannel data, must have length equal to
            number of channels. For single-channel data, defaults to ["channel_0"].
        
        tissue_type (Optional[str]): Type of biological tissue in the image, e.g.,
            "liver", "kidney", etc.
        
        cell_types (Optional[List[str]]): List of cell types expected to be present
            in the image. Used for reference and verification.
        
        image_mpp (Optional[float]): Microns per pixel resolution of the image.
            Used for size-based analysis and visualization scaling.
        
        mask (Optional[np.ndarray]): Ground truth segmentation mask in (1, H, W) format.
            Integer-valued array where 0 is background and positive integers are
            unique cell identifiers.
        
        cell_type_info (Optional[Dict[int, str]]): Mapping from cell identifiers in
            the ground truth mask to their corresponding cell type labels.
        
        predicted_mask (Optional[np.ndarray]): Model-predicted segmentation mask in
            (1, H, W) format. Integer-valued array where 0 is background and
            positive integers are unique cell identifiers.
        
        predicted_cell_types (Optional[Dict[int, str]]): Mapping from cell identifiers
            in the predicted mask to their predicted cell type labels.

    Example:
        >>> image_data = ImageData(
        ...     raw=np.array(...),  # Shape (2, 512, 512)
        ...     image_id="sample_001",
        ...     channel_names=["DAPI", "CD3"],
        ...     tissue_type="lymph_node",
        ...     mask=np.array(...)  # Shape (1, 512, 512)
        ... )
        >>> image_data.validate()  # Verifies data consistency
    """
    raw: np.ndarray
    image_id: Union[int, str]
    channel_names: Optional[List[str]] = None
    tissue_type: Optional[str] = None
    cell_types: Optional[List[str]] = None
    image_mpp: Optional[float] = None
    mask: Optional[np.ndarray] = None
    cell_type_info: Optional[Dict[int, str]] = None
    predicted_mask: Optional[np.ndarray] = None
    predicted_cell_types: Optional[Dict[int, str]] = None

    def __post_init__(self):
        """Standardize data formats after initialization."""
        # Determine if multichannel based on channel names
        is_multichannel = self.channel_names is not None and len(self.channel_names) > 1

        # Standardize raw image format
        self.raw, is_multi = standardize_raw_image(self.raw, is_multichannel)

        # Update channel_names if necessary
        if self.raw is not None and not is_multi and self.channel_names is None:
            self.channel_names = ["channel_0"]

        # Standardize mask formats
        if self.mask is not None:
            self.mask = standardize_mask(self.mask)
        if self.predicted_mask is not None:
            self.predicted_mask = standardize_mask(self.predicted_mask)

    def validate(self) -> None:
        """Validate structural consistency of the image data."""
        if self.raw is None:
            raise ValueError("Raw data cannot be None")

        # Validate raw data shape
        if not (self.raw.ndim == 3):
            raise ValueError("Raw data must have 3 dimensions (C/1, H, W)")

        # Validate channel information
        if self.raw.shape[0] > 1:  # Multichannel
            if self.channel_names is None:
                raise ValueError("Channel names required for multichannel data")
            if len(self.channel_names) != self.raw.shape[0]:
                raise ValueError(
                    "Number of channel names must match number of channels"
                )

        # Validate mask shapes
        if self.mask is not None:
            if self.mask.ndim != 3 or self.mask.shape[0] != 1:
                raise ValueError("Ground truth mask must have shape (1, height, width)")
            if self.mask.shape[1:] != self.raw.shape[1:]:
                raise ValueError("Mask spatial dimensions must match raw data")
            if not np.issubdtype(self.mask.dtype, np.integer):
                raise ValueError("Mask must contain integer values")

        if self.predicted_mask is not None:
            if self.predicted_mask.ndim != 3 or self.predicted_mask.shape[0] != 1:
                raise ValueError("Predicted mask must have shape (1, height, width)")
            if self.predicted_mask.shape[1:] != self.raw.shape[1:]:
                raise ValueError(
                    "Predicted mask spatial dimensions must match raw data"
                )
            if not np.issubdtype(self.predicted_mask.dtype, np.integer):
                raise ValueError("Predicted mask must contain integer values")

            # Check matching dimensions with ground truth if available
            if self.mask is not None and self.predicted_mask.shape != self.mask.shape:
                raise ValueError(
                    "Predicted mask must have same shape as ground truth mask"
                )


