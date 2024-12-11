import zarr
import numpy as np
from typing import List, Dict, Optional, Union, Sequence, Set
from dataclasses import dataclass
import logging
import torch
from pathlib import Path


@dataclass
class ImageData:
    """Container for multichannel microscopy image data and associated metadata.

    This class provides a standardized structure for storing and managing microscopy
    image data along with segmentation masks, cell type annotations, and metadata.
    It supports both ground truth and predicted annotations.

    Attributes:
        raw (np.ndarray): Raw image data with shape (channels, height, width).
            Values are typically fluorescence intensities.
        channel_names (List[str]): Names of imaging channels in order matching raw data
            channels (e.g., ["DAPI", "CD3", "CD20"]).
        tissue_type (str): Type of tissue imaged (e.g., "lymph_node", "tumor").
        cell_types (List[str]): List of expected cell types in the image.
        image_mpp (float): Resolution in microns per pixel.
        file_name (str): Original source file name.
        mask (Optional[np.ndarray]): Ground truth segmentation mask with shape
            (height, width). Each unique integer represents a cell.
        cell_type_info (Optional[Dict[int, str]]): Maps cell indices from mask
            to their ground truth cell types.
        predicted_mask (Optional[np.ndarray]): Model-predicted segmentation mask
            with shape (height, width).
        predicted_cell_types (Optional[Dict[int, str]]): Maps cell indices from
            predicted_mask to their predicted types.
    """

    raw: np.ndarray
    channel_names: List[str]
    tissue_type: str
    cell_types: List[str]
    image_mpp: float
    file_name: str
    mask: Optional[np.ndarray] = None
    cell_type_info: Optional[Dict[int, str]] = None
    predicted_mask: Optional[np.ndarray] = None
    predicted_cell_types: Optional[Dict[int, str]] = None

    def validate(self) -> None:
        """Validate structural consistency of the image data.

        Checks:
        - Raw data has correct shape (channels, height, width)
        - Number of channel names matches number of channels
        - Mask dimensions match raw data dimensions

        Raises:
            ValueError: If any validation check fails
        """
        if self.raw.ndim != 3:
            raise ValueError("Raw data must have shape (channels, height, width)")
        if len(self.channel_names) != self.raw.shape[0]:
            raise ValueError("Number of channel names must match number of channels")
        if self.mask is not None and self.mask.shape != self.raw.shape[1:]:
            raise ValueError("Mask dimensions must match raw data dimensions")

    def to_torch(self) -> torch.Tensor:
        """Convert raw image data to PyTorch tensor.

        Returns:
            torch.Tensor: Raw data as tensor with same shape and dtype
        """
        return torch.from_numpy(self.raw)

    @classmethod
    def from_torch(cls, tensor: torch.Tensor, **kwargs) -> "ImageData":
        """Create ImageData instance from PyTorch tensor.

        Args:
            tensor (torch.Tensor): Tensor with shape (channels, height, width)
            **kwargs: Additional arguments passed to ImageData constructor

        Returns:
            ImageData: New instance with tensor data converted to numpy array
        """
        return cls(raw=tensor.numpy(), **kwargs)


class ZarrDataset:
    """Interface for persistent storage and retrieval of biological image datasets.

    This class provides a standardized interface for storing and loading multichannel
    image data and associated metadata using Zarr storage format. It supports:
    - Hierarchical organization of image data
    - Efficient storage of large datasets
    - Flexible loading of subsets of data
    - Automatic validation of data consistency

    The typical workflow involves:
    1. Loading raw data from an existing dataset
    2. Processing the images (e.g., segmentation, classification)
    3. Saving results to a new dataset

    Example workflow:
        # Load raw data
        raw_dataset = ZarrDataset("raw_data.zarr")
        images = raw_dataset.load_all()

        # Process images
        processed_images = process_images(images)

        # Save processed results
        processed_dataset = ZarrDataset.create("processed_data.zarr",
                                             raw_dataset.get_channel_names())
        processed_dataset.save_all(processed_images)
    """

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

            # Load required data
            raw = group["raw"][:]

            # Load optional data
            mask = group["mask"][0, :] if "mask" in group else None
            predicted_mask = (
                group["predicted_mask"][:] if "predicted_mask" in group else None
            )

            # Load optional cell type info
            cell_type_info = self._load_cell_type_info(group, "cell_type_info")
            predicted_cell_types = self._load_cell_type_info(
                group, "predicted_cell_type_info"
            )

            # Load metadata
            attrs = dict(group.attrs)

            image = ImageData(
                raw=raw,
                channel_names=self.root.attrs["channel_names"],
                tissue_type=attrs.get("tissue_type", "unknown"),
                cell_types=attrs.get("cell_types", []),
                image_mpp=attrs.get("mpp", 0.0),
                file_name=name,
                mask=mask,
                cell_type_info=cell_type_info,
                predicted_mask=predicted_mask,
                predicted_cell_types=predicted_cell_types,
            )
            images.append(image)

        return images

    def load_all(self) -> List[ImageData]:
        """Load all images from dataset.

        Returns:
            List of all ImageData objects in the dataset
        """
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

        # Optional validation
        if validate:
            for image in images:
                image.validate()
                if image.channel_names != self.root.attrs["channel_names"]:
                    raise ValueError(
                        f"Channel names mismatch in {image.file_name}. "
                        f"Expected {self.root.attrs['channel_names']}, "
                        f"got {image.channel_names}"
                    )

        # Track new file names
        new_file_names = []

        for image in images:
            # Create group for this file
            group = self.root.require_group(image.file_name)

            # Save required data
            group.create_dataset("raw", data=image.raw, overwrite=True)

            # Save optional masks if present
            if image.mask is not None:
                group.create_dataset("mask", data=image.mask, overwrite=True)
            if image.predicted_mask is not None:
                group.create_dataset(
                    "predicted_mask", data=image.predicted_mask, overwrite=True
                )

            # Save optional cell type info if present
            if image.cell_type_info is not None:
                self._save_cell_type_info(group, image.cell_type_info, "cell_type_info")
            if image.predicted_cell_types is not None:
                self._save_cell_type_info(
                    group, image.predicted_cell_types, "predicted_cell_type_info"
                )

            # Save metadata
            group.attrs["tissue_type"] = image.tissue_type
            group.attrs["cell_types"] = image.cell_types
            group.attrs["mpp"] = image.image_mpp

            new_file_names.append(image.file_name)

        # Update global file_names attribute
        current_files = set(self.root.attrs["file_names"])
        updated_files = list(current_files.union(new_file_names))
        self.root.attrs["file_names"] = updated_files

    def save_all(self, images: List[ImageData], validate: bool = True) -> None:
        """Save all images to a new dataset.

        Alias for save() to maintain API clarity.
        """
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
