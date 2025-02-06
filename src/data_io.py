import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np
from typing import Tuple, Optional

def standardize_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Standardize mask shape to (B, 1, H, W) format.

    Args:
        mask: Input mask with shape (B, H, W), (B, H, W, 1),
              (B, 1, H, W), or (B, 1, H, W, 1)

    Returns:
        np.ndarray: Standardized mask with shape (B, 1, H, W)

    Raises:
        ValueError: If mask dimensions are invalid
    """
    if mask.ndim == 3:  # (B, H, W)
        return mask[:, np.newaxis, ...]
    elif mask.ndim == 4:
        if mask.shape[1] == 1:  # (B, 1, H, W)
            return mask
        elif mask.shape[-1] == 1:  # (B, H, W, 1)
            return np.transpose(mask, (0, 3, 1, 2))
        else: 
            raise ValueError(
                f"Invalid mask shape {mask.shape}"
            )
    else:
        raise ValueError(
            f"Invalid mask shape {mask.shape}"
        )


def standardize_raw_image(
    raw: np.ndarray
) -> np.ndarray:
    """Standardize raw image shape to (B, C, H, W) format.

    Args:
        raw: Input image array with shape (B, H, W), (B, H, W, C),
             or (B, C, H, W)

    Returns:
        np.ndarray: Standardized array with shape (B, C, H, W)

    Raises:
        ValueError: If input array dimensions are not 3 or 4
    """
    if raw.ndim == 3:  # (B, H, W)
        return raw[:, np.newaxis, ...]
    elif raw.ndim == 4:
        if raw.shape[-1] != min(raw.shape[1:]):  # (B, H, W, C)
            raw = np.transpose(raw, (0, 3, 1, 2))
        return raw
    else:
        raise ValueError(
            f"Invalid raw image shape {raw.shape}"
        )


@dataclass
class ImageData:
    """Container for batched biological image data and associated metadata.
    
    This class provides a standardized structure for storing and managing batched biological
    image data along with related annotations and predictions. Arrays are always stored
    with a batch dimension, even for single images.

    The class standardizes array shapes as follows:
    - Raw images: (B, C, H, W) format where:
        B: batch size
        C: number of channels
        H, W: height and width
    - Masks: (B, 1, H, W) format for both ground truth and predictions

    Attributes:
        raw (np.ndarray): Raw image data in (B, C, H, W) format.
        
        batch_size (int): Number of images in the batch.
        
        image_ids (Union[int, str, List[Union[int, str]]]): Unique identifier(s) for images
            in the batch. Can be a single value for batch size 1, or a list matching 
            batch size.
        
        channel_names (Optional[List[str]]): Names of imaging channels in order matching
            raw data channels. Length must equal number of channels.
        
        tissue_types (Optional[Union[str, List[str]]]): Type of biological tissue for each image,
            e.g., ["liver", "kidney"]. Length must equal batch size.
        
        image_mpps (Optional[Union[float, List[float]]]): Microns per pixel resolution for each
            image. Length must equal batch size.
        
        masks (Optional[np.ndarray]): Ground truth segmentation masks in (B, 1, H, W) 
            format. Integer-valued array where 0 is background and positive integers 
            are unique cell identifiers.
        
        cell_types (Optional[List[Dict[int, str]]]): List of mappings from cell 
            identifiers to cell type labels for each image. Length must equal batch size.
        
        predicted_masks (Optional[np.ndarray]): Model-predicted segmentation masks in
            (B, 1, H, W) format.
        
        predicted_cell_types (Optional[List[Dict[int, str]]]): List of mappings from
            cell identifiers to predicted cell types for each image.
    """
    raw: np.ndarray
    batch_size: int
    image_ids: Union[int, str, List[Union[int, str]]]
    channel_names: Optional[List[str]] = None
    tissue_types: Optional[Union[str, List[str]]] = None
    image_mpps: Optional[List[float]] = None
    masks: Optional[np.ndarray] = None
    cell_types: Optional[List[Dict[int, str]]] = None
    predicted_masks: Optional[np.ndarray] = None
    predicted_cell_types: Optional[List[Dict[int, str]]] = None

    def __post_init__(self):
        """Standardize data formats and validate attributes after initialization."""
        # Standardize array formats
        self.raw = standardize_raw_image(self.raw)
        self.batch_size = self.raw.shape[0]
        
        if self.masks is not None:
            self.masks = standardize_mask(self.masks)
            # Validate mask dimensions match raw image
            if self.masks.shape[0] != self.batch_size or \
            self.masks.shape[2:] != self.raw.shape[2:]:
                raise ValueError(
                    f"Mask shape {self.masks.shape} does not match raw image spatial dimensions "
                    f"(batch_size={self.batch_size}, H={self.raw.shape[2]}, W={self.raw.shape[3]})"
                )
        
        if self.predicted_masks is not None:
            self.predicted_masks = standardize_mask(self.predicted_masks)
            # Validate predicted mask dimensions match raw image
            if self.predicted_masks.shape[0] != self.batch_size or \
            self.predicted_masks.shape[2:] != self.raw.shape[2:]:
                raise ValueError(
                    f"Predicted mask shape {self.predicted_masks.shape} does not match raw image spatial dimensions "
                    f"(batch_size={self.batch_size}, H={self.raw.shape[2]}, W={self.raw.shape[3]})"
                )

        # Validate and standardize image_ids
        if isinstance(self.image_ids, (int, str)):
            if self.batch_size != 1:
                raise ValueError("Single image_id provided but batch size is not 1")
            self.image_ids = [self.image_ids]
        if len(self.image_ids) != self.batch_size:
            raise ValueError(
                f"Number of image_ids ({len(self.image_ids)}) does not match batch size ({self.batch_size})"
            )
        
        # Validate channel_names if provided
        if self.channel_names is not None:
            if len(self.channel_names) != self.raw.shape[1]:
                raise ValueError(
                    f"Number of channel names ({len(self.channel_names)}) does not match "
                    f"number of channels in raw data ({self.raw.shape[1]})"
                )
        
        # Validate and standardize tissue_types
        if self.tissue_types is not None:
            if isinstance(self.tissue_types, str):
                self.tissue_types = [self.tissue_types] * self.batch_size
            if len(self.tissue_types) != self.batch_size:
                raise ValueError(
                    f"Number of tissue types ({len(self.tissue_types)}) does not match batch size ({self.batch_size})"
                )
        
        # Validate and standardize image_mpps
        if self.image_mpps is not None:
            if isinstance(self.image_mpps, (int, float)):
                self.image_mpps = [float(self.image_mpps)] * self.batch_size
            if len(self.image_mpps) != self.batch_size:
                raise ValueError(
                    f"Number of image MPPs ({len(self.image_mpps)}) does not match batch size ({self.batch_size})"
                )
            # Convert all MPPs to float
            self.image_mpps = [float(mpp) for mpp in self.image_mpps]
        
        # Validate cell_types and predicted_cell_types if provided
        if self.cell_types is not None:
            if self.masks is None:
                raise ValueError("Cell types provided but no ground truth masks present")
                
        if self.predicted_cell_types is not None:
            if self.predicted_masks is None:
                raise ValueError("Predicted cell types provided but no predicted masks present")

    @property
    def num_channels(self) -> int:
        """Get the number of channels in the raw image data.
        
        Returns:
            int: Number of channels in dimension 1 of raw data
        """
        return self.raw.shape[1]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Get the spatial dimensions of the images.
        
        Returns:
            Tuple[int, int]: Height and width of the images (H, W)
        """
        return self.raw.shape[2:]