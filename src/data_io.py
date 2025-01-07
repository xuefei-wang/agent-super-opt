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
              (1, height, width) or similar variations

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
    """Container for biological image data and associated metadata."""

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


class ZarrDataset:
    """Interface for persistent storage and retrieval of biological image datasets."""

    def __init__(self, path: str):
        """Initialize dataset for reading.

        Args:
            path: Path to existing zarr store
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Zarr store not found at {path}")
        self.root = zarr.open(path, mode="r")
        if "channel_names" not in self.root.attrs:
            raise ValueError("Invalid zarr store: missing channel_names attribute")

    @classmethod
    def create(cls, path: str, channel_names: List[str]) -> "ZarrDataset":
        """Create a new dataset for writing.

        Args:
            path: Path where zarr store will be created
            channel_names: List of channel names for the dataset

        Returns:
            New ZarrDataset instance in write mode
        """
        if Path(path).exists():
            raise FileExistsError(f"Zarr store already exists at {path}")
        root = zarr.open(path, mode="w")
        root.attrs["channel_names"] = channel_names
        root.attrs["file_names"] = []
        instance = cls.__new__(cls)  # Create without calling __init__
        instance.root = root
        return instance

    def _load_cell_type_info(
        self, group: zarr.Group, dataset_name: str
    ) -> Optional[Dict[int, str]]:
        """Load cell type information from a dataset."""
        if dataset_name not in group:
            return None

        try:
            data = group[dataset_name][:]
            return dict(zip(data["cell_index"], data["cell_type"]))
        except Exception as e:
            logging.warning(f"Failed to load {dataset_name}: {str(e)}")
            return None

    def load(self, file_names: Optional[List[str]] = None) -> List[ImageData]:
        """Load specified images from dataset.

        Args:
            file_names: Optional list of specific files to load. If None,
                loads all files in the dataset.

        Returns:
            List[ImageData]: Loaded image data objects

        Notes:
            - Missing files are skipped with a warning
            - All metadata and available masks are loaded
            - Channel names must match dataset schema
        """
        if file_names is None:
            file_names = list(self.root.attrs["file_names"])

        images = []
        for name in file_names:
            if name not in self.root:
                logging.warning(f"File {name} not found, skipping")
                continue

            group = self.root[name]
            raw = group["raw"][:]
            mask = group["mask"][:] if "mask" in group else None
            predicted_mask = (
                group["predicted_mask"][:] if "predicted_mask" in group else None
            )

            cell_type_info = self._load_cell_type_info(group, "cell_type_info")
            predicted_cell_types = self._load_cell_type_info(
                group, "predicted_cell_type_info"
            )

            attrs = dict(group.attrs)

            image = ImageData(
                raw=raw,
                image_id=name,
                channel_names=self.root.attrs["channel_names"],
                tissue_type=attrs.get("tissue_type", None),
                cell_types=attrs.get("cell_types", None),
                image_mpp=attrs.get("mpp", None),
                mask=mask,
                cell_type_info=cell_type_info,
                predicted_mask=predicted_mask,
                predicted_cell_types=predicted_cell_types,
            )
            images.append(image)

        return images

    def load_all(self) -> List[ImageData]:
        """Load all images from dataset."""
        return self.load()

    def _save_cell_type_info(
        self, group: zarr.Group, cell_type_info: Dict[int, str], dataset_name: str
    ) -> None:
        """Save cell type information to a dataset."""
        cell_indices = list(cell_type_info.keys())
        cell_types = [cell_type_info[idx] for idx in cell_indices]

        info_array = np.zeros(
            len(cell_indices), dtype=[("cell_index", "i4"), ("cell_type", "U60")]
        )
        info_array["cell_index"] = cell_indices
        info_array["cell_type"] = cell_types

        group.create_dataset(dataset_name, data=info_array, overwrite=True)

    def save(self, images: List[ImageData], validate: bool = True) -> None:
        """Save images and associated data to the dataset.

        This method saves both required and optional components:
        - Raw image data (required)
        - Channel information (required)
        - Segmentation masks (optional)
        - Cell type annotations (optional)
        - Image metadata (optional)

        Args:
            images: List of ImageData objects to save
            validate: Whether to validate data consistency before saving

        Raises:
            ValueError: If dataset is read-only or if validation fails

        Notes:
            - Existing data with the same file_name will be overwritten
            - Channel names must match dataset schema when validate=True
            - New file names are automatically tracked in the dataset
        """
        if self.root.read_only:
            raise ValueError("Dataset opened in read-only mode")

        if validate:
            for image in images:
                image.validate()
                if image.channel_names != self.root.attrs["channel_names"]:
                    raise ValueError(
                        f"Channel names mismatch in {image.image_id}. "
                        f"Expected {self.root.attrs['channel_names']}, "
                        f"got {image.channel_names}"
                    )

        new_file_names = []
        for image in images:
            image_id = str(image.image_id)
            group = self.root.require_group(image_id)

            group.create_dataset("raw", data=image.raw, overwrite=True)

            if image.mask is not None:
                group.create_dataset("mask", data=image.mask, overwrite=True)
            if image.predicted_mask is not None:
                group.create_dataset(
                    "predicted_mask", data=image.predicted_mask, overwrite=True
                )

            if image.cell_type_info is not None:
                self._save_cell_type_info(group, image.cell_type_info, "cell_type_info")
            if image.predicted_cell_types is not None:
                self._save_cell_type_info(
                    group, image.predicted_cell_types, "predicted_cell_type_info"
                )

            if image.tissue_type is not None:
                group.attrs["tissue_type"] = image.tissue_type
            if image.cell_types is not None:
                group.attrs["cell_types"] = image.cell_types
            if image.image_mpp is not None:
                group.attrs["mpp"] = image.image_mpp

            new_file_names.append(image_id)

        current_files = set(self.root.attrs["file_names"])
        updated_files = list(current_files.union(new_file_names))
        self.root.attrs["file_names"] = updated_files

    def save_all(self, images: List[ImageData], validate: bool = True) -> None:
        """Save all images to a new dataset."""
        self.save(images, validate=validate)

    def get_channel_names(self) -> List[str]:
        """Get ordered list of channel names in the dataset.

        Returns:
            List[str]: Channel names in order matching the data channels
        """
        return list(self.root.attrs["channel_names"])

    def get_file_names(self) -> List[str]:
        """Get list of all image files in the dataset.

        Returns:
            List[str]: Names of all image files in the dataset
        """
        return list(self.root.attrs["file_names"])


class NpzDataset:
    """Interface for loading and saving single-channel data from npz files.

    This class provides a standardized interface for handling datasets stored in
    numpy's .npz format, specifically structured for single-channel images and
    their corresponding masks.

    The data is expected to be stored with keys:
    - 'X': Raw images with shape (batch, height, width, 1) or (batch, height, width)
    - 'y': Masks with shape (batch, height, width, 1) or (batch, height, width)
    - 'meta': Metadata array where first column contains image IDs
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
    def _standardize_array(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Standardize array shape to (batch, channel, height, width) format.
        
        Args:
            arr: Input array with shape (batch, height, width, 1) or (batch, height, width)
            is_mask: Whether the array is a mask (affects channel handling)
            
        Returns:
            np.ndarray: Standardized array with shape (batch, channel, height, width)
            
        Raises:
            ValueError: If input shape is invalid
        """
        if arr.ndim not in [3, 4]:
            raise ValueError(f"Invalid array dimensionality {arr.ndim}. Expected 3 or 4 dimensions.")
            
        # Handle (batch, H, W) -> (batch, 1, H, W)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
            
        # Now arr is (batch, H, W, C)
        # Convert to (batch, C, H, W)
        arr = np.transpose(arr, (0, 3, 1, 2))
        
        # For masks, ensure single channel and integer type
        if is_mask:
            if arr.shape[1] != 1:
                raise ValueError("Masks must have exactly one channel")
            arr = arr.astype(np.int32)
            
        return arr

    @staticmethod
    def _validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
        """Validate shapes and dimensions of input arrays."""
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

    def load(self, indices: Optional[List[int]] = None) -> List[ImageData]:
        """Load specified images from dataset.

        Args:
            indices: Optional list of indices to load. If None,
                loads all images in the dataset.

        Returns:
            List[ImageData]: Loaded image data objects

        Raises:
            IndexError: If any index is out of bounds
        """
        with np.load(self.path, allow_pickle=True) as data:
            X = data["X"]  # (batch, H, W, 1) or (batch, H, W)
            y = data["y"]  # (batch, H, W, 1) or (batch, H, W)
            
            # Standardize shapes
            X = self._standardize_array(X, is_mask=False)  # -> (batch, C, H, W)
            y = self._standardize_array(y, is_mask=True)   # -> (batch, 1, H, W)

            if indices is None:
                indices = range(len(X))

            try:
                images = []
                for i in indices:
                    images.append(
                        ImageData(
                            raw=X[i],        # (C, H, W)
                            mask=y[i],       # (1, H, W)
                            image_id=i,
                            channel_names=["channel_0"]  # Single channel data
                        )
                    )
                return images
            except IndexError as e:
                raise IndexError(f"Invalid index in dataset: {str(e)}")

    def load_all(self) -> List[ImageData]:
        """Load all images from dataset."""
        return self.load()

    @classmethod
    def save(
        cls, path: Union[str, Path], images: List[ImageData], validate: bool = True
    ) -> None:
        """Save images and associated data to a new dataset.

        Args:
            path: Path where .npz file will be created
            images: List of ImageData objects to save
            validate: Whether to validate data consistency before saving

        Raises:
            ValueError: If validation fails or data format is invalid
            FileExistsError: If file already exists
            
        Notes:
            Saves data in (batch, H, W, 1) format to match the expected input format.
            Single channel data is enforced for both raw images and masks.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Dataset already exists at {path}")

        if validate:
            for image in images:
                image.validate()
                # Ensure single channel data
                if image.raw.shape[0] != 1:
                    raise ValueError(f"Raw data must be single channel, got {image.raw.shape[0]} channels")

        # Stack data - ImageData objects have standardized shapes (C, H, W)
        X = np.stack([img.raw for img in images])  # (batch, C, H, W)
        y = np.stack([img.mask for img in images])  # (batch, 1, H, W)

        # Convert to required output format (batch, H, W, 1)
        X = np.transpose(X, (0, 2, 3, 1))  # -> (batch, H, W, C)
        y = np.transpose(y, (0, 2, 3, 1))  # -> (batch, H, W, 1)

        # Squeeze any extra dimensions but keep the channel dimension
        X = np.expand_dims(X.squeeze(), axis=-1)  # Ensure (batch, H, W, 1)
        y = np.expand_dims(y.squeeze(), axis=-1)  # Ensure (batch, H, W, 1)

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
        """
        with np.load(self.path, allow_pickle=True) as data:
            return data["X"].shape, data["y"].shape

    def get_image_ids(self) -> List[int]:
        """Get list of all image IDs in the dataset.

        Returns:
            List[int]: Image IDs in order matching the data (0-based indices)
        """
        with np.load(self.path) as data:
            return list(range(len(data["X"])))

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        with np.load(self.path) as data:
            return len(data["X"])