"""
Module for loading and saving biological image data in various formats.
Supports both Zarr and NPZ data formats with proper batch handling.
"""

import zarr
import numpy as np
from pathlib import Path
import logging
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
            if self.masks.shape[0] != self.batch_size or \
            self.masks.shape[2:] != self.raw.shape[2:]:
                raise ValueError(
                    f"Mask shape {self.masks.shape} does not match raw image spatial dimensions "
                    f"(batch_size={self.batch_size}, H={self.raw.shape[2]}, W={self.raw.shape[3]})"
                )
        
        if self.predicted_masks is not None:
            self.predicted_masks = standardize_mask(self.predicted_masks)
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
        """Get the number of channels in the raw image data."""
        return self.raw.shape[1]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Get the spatial dimensions (H, W) of the images."""
        return self.raw.shape[2:]


class ZarrDataset:
    """Interface for persistent storage and retrieval of biological image datasets.
    
    Provides methods for saving and loading image data in Zarr format, which is 
    optimized for large array storage. Supports multi-channel images, masks, 
    and associated metadata. All loaded data follows the batched ImageData format.
    """

    def __init__(self, path: str):
        """Initialize dataset for reading."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Zarr store not found at {path}")
        self.root = zarr.open(path, mode="r")
        if "channel_names" not in self.root.attrs:
            raise ValueError("Invalid zarr store: missing channel_names attribute")

    @classmethod
    def create(cls, path: str, channel_names: List[str]) -> "ZarrDataset":
        """Create a new dataset for writing."""
        if Path(path).exists():
            raise FileExistsError(f"Zarr store already exists at {path}")
        root = zarr.open(path, mode="w")
        root.attrs["channel_names"] = channel_names
        root.attrs["file_names"] = []
        instance = cls.__new__(cls)
        instance.root = root
        return instance

    def _load_cell_type_info(
        self, group: zarr.Group, dataset_name: str
    ) -> Optional[List[Dict[int, str]]]:
        """Load cell type information from a dataset."""
        if dataset_name not in group:
            return None

        try:
            data = group[dataset_name][:]
            batch_size = len(data)
            cell_type_dicts = []
            for i in range(batch_size):
                cell_type_dicts.append(
                    dict(zip(data[i]["cell_index"], data[i]["cell_type"]))
                )
            return cell_type_dicts
        except Exception as e:
            logging.warning(f"Failed to load {dataset_name}: {str(e)}")
            return None

    def load(self, file_names: Optional[List[str]] = None) -> ImageData:
        """Load specified images from dataset into a single batched ImageData object."""
        if file_names is None:
            file_names = list(self.root.attrs["file_names"])

        raw_list = []
        mask_list = []
        pred_mask_list = []
        tissue_types = []
        image_mpps = []
        cell_types_list = []
        cell_type_info_list = []
        pred_cell_types_list = []

        for name in file_names:
            if name not in self.root:
                logging.warning(f"File {name} not found, skipping")
                continue

            group = self.root[name]
            raw = standardize_raw_image(group["raw"][:])
            raw_list.append(raw)

            if "mask" in group:
                mask = standardize_mask(group["mask"][:])
                mask_list.append(mask)

            if "predicted_mask" in group:
                pred_mask = standardize_mask(group["predicted_mask"][:])
                pred_mask_list.append(pred_mask)

            attrs = dict(group.attrs)
            tissue_types.append(attrs.get("tissue_type", None))
            image_mpps.append(attrs.get("mpp", None))
            cell_types_list.append(attrs.get("cell_types", None))

            cell_type_info = self._load_cell_type_info(group, "cell_type_info")
            if cell_type_info:
                cell_type_info_list.extend(cell_type_info)

            pred_cell_types = self._load_cell_type_info(group, "predicted_cell_type_info")
            if pred_cell_types:
                pred_cell_types_list.extend(pred_cell_types)

        # Stack arrays along batch dimension
        raw_batch = np.stack(raw_list, axis=0)
        masks_batch = np.stack(mask_list, axis=0) if mask_list else None
        pred_masks_batch = np.stack(pred_mask_list, axis=0) if pred_mask_list else None

        return ImageData(
            raw=raw_batch,
            batch_size=len(file_names),
            image_ids=file_names,
            channel_names=self.root.attrs["channel_names"],
            tissue_types=tissue_types if any(t is not None for t in tissue_types) else None,
            image_mpps=image_mpps if any(m is not None for m in image_mpps) else None,
            masks=masks_batch,
            cell_types=cell_types_list if cell_types_list else None,
            predicted_masks=pred_masks_batch,
            predicted_cell_types=pred_cell_types_list if pred_cell_types_list else None
        )

    def load_all(self) -> ImageData:
        """Load all images from dataset into a single batched ImageData object."""
        return self.load()

    def _save_cell_type_info(
        self, group: zarr.Group, 
        cell_type_info: List[Dict[int, str]], 
        dataset_name: str
    ) -> None:
        """Save cell type information to a dataset."""
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

    def save(self, image_data: ImageData, validate: bool = True) -> None:
        """Save batched ImageData to the dataset."""
        if self.root.read_only:
            raise ValueError("Dataset opened in read-only mode")

        if validate:
            image_data.validate()
            if image_data.channel_names != self.root.attrs["channel_names"]:
                raise ValueError(
                    "Channel names mismatch. "
                    f"Expected {self.root.attrs['channel_names']}, "
                    f"got {image_data.channel_names}"
                )

        new_file_names = []
        for i in range(image_data.batch_size):
            image_id = str(image_data.image_ids[i])
            group = self.root.require_group(image_id)

            # Save raw data for this batch item
            group.create_dataset("raw", data=image_data.raw[i], overwrite=True)

            if image_data.masks is not None:
                group.create_dataset("mask", data=image_data.masks[i], overwrite=True)
            
            if image_data.predicted_masks is not None:
                group.create_dataset(
                    "predicted_mask", data=image_data.predicted_masks[i], overwrite=True
                )

            if image_data.cell_types is not None:
                self._save_cell_type_info(
                    group, image_data.cell_types[i:i+1], "cell_type_info"
                )
            
            if image_data.predicted_cell_types is not None:
                self._save_cell_type_info(
                    group, image_data.predicted_cell_types[i:i+1], "predicted_cell_type_info"
                )

            # Save metadata
            if image_data.tissue_types is not None:
                group.attrs["tissue_type"] = image_data.tissue_types[i]
                
            if image_data.cell_types is not None:
                group.attrs["cell_types"] = image_data.cell_types[i]
                
            if image_data.image_mpps is not None:
                group.attrs["mpp"] = image_data.image_mpps[i]

            new_file_names.append(image_id)

        current_files = set(self.root.attrs["file_names"])
        updated_files = list(current_files.union(new_file_names))
        self.root.attrs["file_names"] = updated_files

    def save_all(self, image_data: ImageData, validate: bool = True) -> None:
        """Save all images from a batched ImageData object."""
        self.save(image_data, validate=validate)

    def get_channel_names(self) -> List[str]:
        """Get ordered list of channel names in the dataset."""
        return list(self.root.attrs["channel_names"])

    def get_file_names(self) -> List[str]:
        """Get list of all image files in the dataset."""
        return list(self.root.attrs["file_names"])


class NpzDataset:
    """Interface for loading and saving single-channel data from npz files.

    This class provides a standardized interface for handling datasets stored in
    numpy's .npz format, specifically structured for single-channel images and
    their corresponding masks.

    The data is expected to be stored with keys:
    - 'X': Raw images with shape (B, H, W, 1) or (B, H, W)
    - 'y': Masks with shape (B, H, W, 1) or (B, H, W)
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize dataset for reading.

        Args:
            path: Path to existing .npz file

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
            raise ValueError("Raw and mask data must have matching batch sizes")

        # Check spatial dimensions match
        if x_spatial != y_spatial:
            raise ValueError("Raw and mask data must have matching spatial dimensions")

    def load(self, indices: Optional[List[int]] = None) -> ImageData:
        """Load specified images from dataset into a single batched ImageData object.

        Args:
            indices: Optional list of indices to load. If None,
                loads all images in the dataset.

        Returns:
            ImageData: Batched image data object

        Raises:
            IndexError: If any index is out of bounds
        """
        with np.load(self.path, allow_pickle=True) as data:
            X = data["X"]  # (B, H, W, 1) or (B, H, W)
            y = data["y"]  # (B, H, W, 1) or (B, H, W)

            if indices is None:
                indices = range(len(X))

            try:
                # Extract selected indices
                X_batch = X[indices]
                y_batch = y[indices]

                # Standardize shapes to (B, C, H, W) format
                X_batch = standardize_raw_image(X_batch)
                y_batch = standardize_mask(y_batch)

                return ImageData(
                    raw=X_batch,
                    batch_size=len(indices),
                    image_ids=indices,
                    channel_names=["channel_0"],  # Single channel data
                    masks=y_batch
                )
            except IndexError as e:
                raise IndexError(f"Invalid index in dataset: {str(e)}")

    def load_all(self) -> ImageData:
        """Load all images from dataset into a single batched ImageData object."""
        return self.load()

    @classmethod
    def save(cls, path: Union[str, Path], image_data: ImageData, validate: bool = True) -> None:
        """Save batched ImageData to a new dataset.

        Args:
            path: Path where .npz file will be created
            image_data: ImageData object containing batch of images to save
            validate: Whether to validate data consistency before saving

        Raises:
            ValueError: If validation fails or data format is invalid
            FileExistsError: If file already exists
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Dataset already exists at {path}")

        if validate:
            image_data.validate()
            # Ensure single channel data
            if image_data.raw.shape[1] != 1:
                raise ValueError(
                    f"Raw data must be single channel, got {image_data.raw.shape[1]} channels"
                )

        # Convert from (B, C, H, W) to required output format (B, H, W, 1)
        X = np.transpose(image_data.raw, (0, 2, 3, 1))  # -> (B, H, W, C)
        y = np.transpose(image_data.masks, (0, 2, 3, 1)) if image_data.masks is not None else None

        cls._validate_shapes(X, y)

        # Save to npz file
        try:
            np.savez_compressed(path, X=X, y=y)
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {str(e)}")

    def get_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Get shapes of the raw and mask data.
        
        Returns:
            Tuple containing shapes of (X, y) arrays in their original format
            The first shape corresponds to raw data (B, H, W, 1)
            The second shape corresponds to mask data (B, H, W, 1)
        """
        with np.load(self.path, allow_pickle=True) as data:
            return data["X"].shape, data["y"].shape

    def get_image_ids(self) -> List[int]:
        """Get list of all image IDs in the dataset.
        
        Returns:
            List of sequential integers from 0 to n-1 where n is dataset size
        """
        with np.load(self.path) as data:
            return list(range(len(data["X"])))

    def __len__(self) -> int:
        """Get number of samples in the dataset.
        
        Returns:
            Integer count of images in the dataset
        """
        with np.load(self.path) as data:
            return len(data["X"])