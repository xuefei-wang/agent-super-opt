"""
Module for loading and saving biological image data in various formats.
Provides framework-agnostic data structures and IO interfaces.
"""

import zarr
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
            raise ValueError(f"Invalid mask shape {mask.shape}")
    else:
        raise ValueError(f"Invalid mask shape {mask.shape}")

def standardize_raw_image(raw: np.ndarray) -> np.ndarray:
    """Standardize raw image shape to (B, H, W, C) format.

    Args:
        raw: Input image array with shape (B, H, W), (B, H, W, C),
             or (B, C, H, W)

    Returns:
        np.ndarray: Standardized array with shape (B, H, W, C)

    Raises:
        ValueError: If input array dimensions are not 3 or 4
    """
    if raw.ndim == 3:  # (B, H, W)
        return raw[:, :, :, np.newaxis]
    elif raw.ndim == 4:
        if raw.shape[1] == min(raw.shape[1:]):  # (B, C, H, W)
            raw = np.transpose(raw, (0, 2, 3, 1))
        return raw
    else:
        raise ValueError(f"Invalid raw image shape {raw.shape}")

@dataclass
class ImageData:
    """Framework-agnostic container for batched biological image data.
    
    This class provides a standardized structure for storing and managing batched 
    biological image data along with related annotations and predictions.
    All data is stored as numpy arrays for framework independence.

    Arrays are standardized to the following formats:
    - Raw images: (B, H, W, C) where:
        B: batch size
        H, W: height and width
        C: number of channels
    - Masks: (B, 1, H, W) for both ground truth and predictions

    Attributes:
        raw (np.ndarray): Raw image data in (B, H, W, C) format.
        
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
        """Validate and standardize data after initialization."""
        self.batch_size = self.raw.shape[0]
        
        if self.masks is not None:
            if self.masks.shape[0] != self.batch_size or \
               self.masks.shape[2:] != self.raw.shape[2:]:
                raise ValueError(
                    f"Mask shape {self.masks.shape} does not match raw image dimensions "
                    f"(batch_size={self.batch_size}, H={self.raw.shape[2]}, W={self.raw.shape[3]})"
                )
        
        if self.predicted_masks is not None:
            if self.predicted_masks.shape[0] != self.batch_size or \
               self.predicted_masks.shape[2:] != self.raw.shape[2:]:
                raise ValueError(
                    f"Predicted mask shape {self.predicted_masks.shape} does not match dimensions "
                    f"(batch_size={self.batch_size}, H={self.raw.shape[2]}, W={self.raw.shape[3]})"
                )

        # Standardize image_ids
        if isinstance(self.image_ids, (int, str)):
            if self.batch_size != 1:
                raise ValueError("Single image_id provided but batch size is not 1")
            self.image_ids = [self.image_ids]
        if len(self.image_ids) != self.batch_size:
            raise ValueError(
                f"Number of image_ids ({len(self.image_ids)}) does not match batch size ({self.batch_size})"
            )
        
        # Validate and standardize metadata
        if self.channel_names is not None and len(self.channel_names) != self.raw.shape[1]:
            raise ValueError(
                f"Number of channel names ({len(self.channel_names)}) does not match "
                f"number of channels ({self.raw.shape[1]})"
            )
        
        if self.tissue_types is not None:
            if isinstance(self.tissue_types, str):
                self.tissue_types = [self.tissue_types] * self.batch_size
            if len(self.tissue_types) != self.batch_size:
                raise ValueError(
                    f"Number of tissue types ({len(self.tissue_types)}) does not match batch size"
                )
        
        if self.image_mpps is not None:
            if isinstance(self.image_mpps, (int, float)):
                self.image_mpps = [float(self.image_mpps)] * self.batch_size
            if len(self.image_mpps) != self.batch_size:
                raise ValueError(
                    f"Number of image MPPs ({len(self.image_mpps)}) does not match batch size"
                )
            self.image_mpps = [float(mpp) for mpp in self.image_mpps]

    @property
    def num_channels(self) -> int:
        """Get number of channels in raw image data."""
        return self.raw.shape[1]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Get spatial dimensions (H, W) of images."""
        return self.raw.shape[2:]
    
    def get_item(self, idx: int) -> 'ImageData':
        """Get single item from batch as new ImageData object.
        
        Args:
            idx: Index in batch to retrieve
            
        Returns:
            New ImageData object containing single item
        """
        if idx >= self.batch_size:
            raise IndexError(f"Index {idx} out of range for batch size {self.batch_size}")
            
        return ImageData(
            raw=self.raw[idx:idx+1],
            batch_size=1,
            image_ids=[self.image_ids[idx]],
            channel_names=self.channel_names,
            tissue_types=[self.tissue_types[idx]] if self.tissue_types else None,
            image_mpps=[self.image_mpps[idx]] if self.image_mpps else None,
            masks=self.masks[idx:idx+1] if self.masks is not None else None,
            cell_types=[self.cell_types[idx]] if self.cell_types else None,
            predicted_masks=self.predicted_masks[idx:idx+1] if self.predicted_masks is not None else None,
            predicted_cell_types=[self.predicted_cell_types[idx]] if self.predicted_cell_types else None
        )
    
    def __len__(self) -> int:
        """Get batch size."""
        return self.batch_size
    
    def __iter__(self) -> Iterator['ImageData']:
        """Iterate over items in batch."""
        for i in range(self.batch_size):
            yield self.get_item(i)


class BaseDataset(ABC):
    """Abstract base class for dataset IO interfaces."""
    
    @abstractmethod
    def load(self, indices: Optional[List[int]] = None) -> ImageData:
        """Load specified indices into ImageData object."""
        pass
    
    @abstractmethod
    def save(self, image_data: ImageData) -> None:
        """Save ImageData object to storage."""
        pass
    
    @abstractmethod
    def get_image_ids(self) -> List[Union[int, str]]:
        """Get list of all image IDs in dataset."""
        pass
    
    def __len__(self) -> int:
        """Get number of images in dataset."""
        return len(self.get_image_ids())


class ZarrDataset(BaseDataset):
    """Dataset interface for Zarr storage format.
    
    Provides efficient access to large-scale image datasets stored in Zarr format.
    Supports multi-channel images, masks, and rich metadata.
    """
    
    def __init__(self, path: Union[str, Path]):
        """Initialize dataset from existing Zarr store.
        
        Args:
            path: Path to Zarr store directory
        
        Raises:
            FileNotFoundError: If Zarr store doesn't exist
            ValueError: If store is missing required attributes
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Zarr store not found at {path}")
            
        self.root = zarr.open(path, mode='r')
        if "channel_names" not in self.root.attrs:
            raise ValueError("Invalid Zarr store: missing channel_names attribute")
            
        self.channel_names = list(self.root.attrs["channel_names"])
        self.file_names = list(self.root.attrs.get("file_names", []))
    
    @classmethod
    def create(cls, path: str, channel_names: List[str]) -> 'ZarrDataset':
        """Create new Zarr dataset.
        
        Args:
            path: Path where Zarr store will be created
            channel_names: Names of image channels
            
        Returns:
            New ZarrDataset instance
        """
        if Path(path).exists():
            raise FileExistsError(f"Zarr store already exists at {path}")
            
        root = zarr.open(path, mode='w')
        root.attrs["channel_names"] = channel_names
        root.attrs["file_names"] = []
        
        instance = cls(path)
        instance.root = root
        return instance
    
    def _load_cell_type_info(
        self, group: zarr.Group, dataset_name: str
    ) -> Optional[List[Dict[int, str]]]:
        """Load cell type information from dataset.
        
        Args:
            group: Zarr group containing the dataset
            dataset_name: Name of dataset containing cell type info
            
        Returns:
            List of cell type dictionaries, or None if not available
        """
        if dataset_name not in group:
            return None
            
        try:
            data = group[dataset_name][:]
            batch_size = len(data)
            return [
                dict(zip(data[i]["cell_index"], data[i]["cell_type"]))
                for i in range(batch_size)
            ]
        except Exception as e:
            logging.warning(f"Failed to load {dataset_name}: {str(e)}")
            return None
    
    def _save_cell_type_info(
        self, group: zarr.Group,
        cell_type_info: List[Dict[int, str]],
        dataset_name: str
    ) -> None:
        """Save cell type information to dataset.
        
        Args:
            group: Zarr group to save to
            cell_type_info: List of cell type dictionaries
            dataset_name: Name for the dataset
        """
        batch_size = len(cell_type_info)
        max_cells = max(len(d) for d in cell_type_info)
        
        info_array = np.zeros(
            (batch_size, max_cells),
            dtype=[("cell_index", "i4"), ("cell_type", "U60")]
        )
        
        for i, cell_dict in enumerate(cell_type_info):
            cell_indices = list(cell_dict.keys())
            cell_types = [cell_dict[idx] for idx in cell_indices]
            info_array[i]["cell_index"][:len(cell_indices)] = cell_indices
            info_array[i]["cell_type"][:len(cell_types)] = cell_types
            
        group.create_dataset(dataset_name, data=info_array, overwrite=True)
    
    def load(self, file_names: Optional[List[str]] = None) -> ImageData:
        """Load specified images into ImageData object.
        
        Args:
            file_names: List of file names to load, or None for all files
            
        Returns:
            ImageData object containing loaded data
        """
        if file_names is None:
            file_names = self.file_names
            
        raw_list = []
        mask_list = []
        pred_mask_list = []
        tissue_types = []
        image_mpps = []
        cell_types_list = []
        pred_cell_types_list = []
        
        for name in file_names:
            if name not in self.root:
                logging.warning(f"File {name} not found, skipping")
                continue
                
            group = self.root[name]
            
            # Load raw data and masks
            raw = standardize_raw_image(group["raw"][:])
            raw_list.append(raw)
            
            if "mask" in group:
                mask = standardize_mask(group["mask"][:])
                mask_list.append(mask)
                
            if "predicted_mask" in group:
                pred_mask = standardize_mask(group["predicted_mask"][:])
                pred_mask_list.append(pred_mask)
            
            # Load metadata
            attrs = dict(group.attrs)
            tissue_types.append(attrs.get("tissue_type"))
            image_mpps.append(attrs.get("mpp"))
            
            # Load cell type information
            cell_type_info = self._load_cell_type_info(group, "cell_type_info")
            if cell_type_info:
                cell_types_list.extend(cell_type_info)
                
            pred_cell_types = self._load_cell_type_info(group, "predicted_cell_type_info")
            if pred_cell_types:
                pred_cell_types_list.extend(pred_cell_types)
        
        # Stack arrays
        raw_batch = np.stack(raw_list, axis=0)
        masks_batch = np.stack(mask_list, axis=0) if mask_list else None
        pred_masks_batch = np.stack(pred_mask_list, axis=0) if pred_mask_list else None
        
        return ImageData(
            raw=standardize_raw_image(raw_batch),
            batch_size=len(file_names),
            image_ids=file_names,
            channel_names=self.channel_names,
            tissue_types=tissue_types if any(t is not None for t in tissue_types) else None,
            image_mpps=image_mpps if any(m is not None for m in image_mpps) else None,
            masks=standardize_mask(masks_batch),
            cell_types=cell_types_list if cell_types_list else None,
            predicted_masks=standardize_mask(pred_masks_batch),
            predicted_cell_types=pred_cell_types_list if pred_cell_types_list else None
        )
    
    def save(self, image_data: ImageData) -> None:
        """Save ImageData object to Zarr store.
        
        Args:
            image_data: ImageData object to save
        """
        if self.root.read_only:
            raise ValueError("Dataset opened in read-only mode")
            
        # Validate channel names
        if image_data.channel_names != self.channel_names:
            raise ValueError(
                f"Channel names mismatch. Expected {self.channel_names}, "
                f"got {image_data.channel_names}"
            )
        
        # Save each item in batch
        new_file_names = []
        for i in range(image_data.batch_size):
            image_id = str(image_data.image_ids[i])
            group = self.root.require_group(image_id)
            
            # Save raw data
            group.create_dataset("raw", data=image_data.raw[i], overwrite=True)
            
            # Save masks if present
            if image_data.masks is not None:
                group.create_dataset("mask", data=image_data.masks[i], overwrite=True)
                
            if image_data.predicted_masks is not None:
                group.create_dataset(
                    "predicted_mask",
                    data=image_data.predicted_masks[i],
                    overwrite=True
                )
            
            # Save cell type information
            if image_data.cell_types is not None:
                self._save_cell_type_info(
                    group,
                    image_data.cell_types[i:i+1],
                    "cell_type_info"
                )
                
            if image_data.predicted_cell_types is not None:
                self._save_cell_type_info(
                    group,
                    image_data.predicted_cell_types[i:i+1],
                    "predicted_cell_type_info"
                )
            
            # Save metadata
            if image_data.tissue_types is not None:
                group.attrs["tissue_type"] = image_data.tissue_types[i]
                
            if image_data.image_mpps is not None:
                group.attrs["mpp"] = image_data.image_mpps[i]
            
            new_file_names.append(image_id)
        
        # Update file list
        current_files = set(self.file_names)
        self.file_names = list(current_files.union(new_file_names))
        self.root.attrs["file_names"] = self.file_names
    
    def get_image_ids(self) -> List[str]:
        """Get list of all image IDs in dataset."""
        return self.file_names


class NpzDataset(BaseDataset):
    """Dataset interface for NPZ file format."""

    def __init__(self, path: Union[str, Path]):
        """Initialize dataset from NPZ file.
        
        Args:
            path: Path to .npz file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
            
        try:
            with np.load(self.path) as data:
                if not all(key in data.files for key in ["X", "y"]):
                    raise ValueError("Missing required keys in dataset")
                self._validate_shapes(data["X"], data["y"])
        except Exception as e:
            raise ValueError(f"Invalid dataset format: {str(e)}")
    
    @staticmethod
    def _validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
        """Validate shapes and dimensions of input arrays.
        
        Args:
            X: Raw image data array
            y: Mask data array
            
        Raises:
            ValueError: If dimensions don't match or are invalid
        """
        # Check basic dimensionality
        if X.ndim not in [3, 4] or y.ndim not in [3, 4]:
            raise ValueError("Arrays must have 3 or 4 dimensions")

        # Get spatial dimensions (height, width)
        x_spatial = X.shape[1:3] if X.ndim == 3 else X.shape[1:3]
        y_spatial = y.shape[1:3] if y.ndim == 3 else y.shape[1:3]

        # Check batch size matches
        if X.shape[0] != y.shape[0]:
            raise ValueError("Raw and mask arrays must have same batch size")

        # Check spatial dimensions match
        if x_spatial != y_spatial:
            raise ValueError("Raw and mask arrays must have same spatial dimensions")
    
    def load(self, indices: Optional[List[int]] = None) -> ImageData:
        """Load specified indices into ImageData object.

        Args:
            indices: Optional list of indices to load. If None,
                loads all images in the dataset.

        Returns:
            ImageData: Batched image data object

        Raises:
            IndexError: If any index is out of bounds
        """
        with np.load(self.path) as data:
            X = data["X"]  # (B, H, W, 1) or (B, H, W)
            y = data["y"]  # (B, H, W, 1) or (B, H, W)
            
            if indices is None:
                indices = range(len(X))
                
            try:
                X_batch = X[indices]
                y_batch = y[indices]
                
                return ImageData(
                    raw=standardize_raw_image(X_batch),
                    batch_size=len(indices),
                    image_ids=indices,
                    channel_names=["channel_0"],
                    masks=standardize_mask(y_batch)
                )
            except IndexError as e:
                raise IndexError(f"Invalid index in dataset: {str(e)}")
    
    def save(self, image_data: ImageData) -> None:
        """Save ImageData object to NPZ file.
        
        Args:
        
            image_data: ImageData object to save
            
        Raises:
            FileExistsError: If dataset already exists at save path
            RuntimeError: If save operation fails
        """
        if self.path.exists():
            raise FileExistsError(f"Dataset already exists at {self.path}")
            
        X = np.transpose(image_data.raw, (0, 2, 3, 1))
        y = np.transpose(image_data.masks, (0, 2, 3, 1)) if image_data.masks is not None else None
        
        self._validate_shapes(X, y)
        
        try:
            np.savez_compressed(self.path, X=X, y=y)
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {str(e)}")
    
    def get_image_ids(self) -> List[int]:
        """Get list of sequential image IDs."""
        with np.load(self.path) as data:
            return list(range(len(data["X"])))

