"""
Cell Phenotyping Framework: A flexible system for cell type prediction.

This module provides a comprehensive framework for implementing and running cell
phenotyping models in biological imaging. It includes:

- Base classes defining standard interfaces for phenotyping models
- Configuration management for model parameters
- Built-in implementation of the DeepCell Types phenotyping model
- Support for preprocessing, prediction, and evaluation pipelines

The framework is designed to be extensible, allowing new phenotyping models to be
added while maintaining a consistent interface.

Example usage:
    ```python
    from phenotyping import PhenotyperDeepCellTypes
    from data_io import ImageData
    
    # Initialize with custom configuration
    config = DeepCellTypesConfig(
        model_path='path/to/weights.pt',
        batch_size=512
    )
    phenotyper = PhenotyperDeepCellTypes(config)
    
    # Run prediction pipeline
    phenotyper.run_pipeline(
        images=[image1, image2],
        celltype_mapping={'CD3+': 'T cell', 'CD20+': 'B cell'}
    )
    
    # Evaluate results
    metrics = phenotyper.calculate_metrics([image1, image2])
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Sequence, Union
from collections import defaultdict
import zarr
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .deepcell_types.model import CellTypeCLIPModel
from .deepcell_types.dataset import PatchDataset
from .deepcell_types.deepcelltypes_kit.config import DCTConfig
from .deepcell_types.deepcelltypes_kit.image_funcs import patch_generator
from .deepcell_types.predict import BatchData
from .data_io import ImageData, standardize_mask, standardize_raw_image


@dataclass
class PhenotyperConfig:
    """Base configuration for phenotyping models.

    This class defines the common configuration parameters that all phenotyping models
    should support. Specific model implementations can extend this with additional
    parameters.

    Attributes:
        model_path: Path to the model weights file
        device: Device to run the model on ('cuda' or 'cpu')
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        intermediate_dir: Directory for storing intermediate files
    """

    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 24
    intermediate_dir: str = "artifacts"


class BasePhenotyper(ABC):
    """Abstract base class for cell phenotyping models.

    This class defines the interface that all phenotyping models must implement.
    It provides a common structure for data preprocessing, prediction, and result handling.

    New phenotyping models should inherit from this class and implement all abstract methods.
    """

    def __init__(self, config: PhenotyperConfig):
        """Initialize base phenotyper.

        Args:
            config: Configuration object for the phenotyper
        """
        self.config = config
        self._create_directories()
        self.model = self._initialize_model()

    def _create_directories(self) -> None:
        """Create necessary directories for intermediate files."""
        Path(self.config.intermediate_dir).mkdir(exist_ok=True)

    @abstractmethod
    def _initialize_model(self) -> Any:
        """Initialize and return the phenotyping model.

        Returns:
            Any: The initialized model instance

        Raises:
            RuntimeError: If model initialization fails
        """
        pass

    @abstractmethod
    def preprocess(self, data: Any, **kwargs) -> Any:
        """Preprocess raw data for model input.

        Args:
            data: Raw data to preprocess
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed data ready for model input

        Raises:
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def predict(
        self, preprocessed_data: Any, images: Sequence[ImageData], **kwargs
    ) -> List[ImageData]:
        """Generate predictions from preprocessed data.

        Args:
            preprocessed_data: Data that has been through preprocessing
            images: Original ImageData objects to update with predictions
            **kwargs: Additional prediction parameters

        Returns:
            List[ImageData]: Original images updated with predictions

        Raises:
            RuntimeError: If prediction fails
        """
        pass

    @staticmethod
    def calculate_metrics(images: Sequence[ImageData]) -> Dict[str, float]:
        """Calculate classification metrics comparing predicted vs true cell types.

        Args:
            images: List of ImageData objects containing both ground truth
                (cell_type_info) and predictions (predicted_cell_types)

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics:
                - accuracy: Overall classification accuracy
                - precision_macro: Unweighted mean of precision per class
                - recall_macro: Unweighted mean of recall per class
                - f1_macro: Unweighted mean of F1 score per class
                - precision_weighted: Precision weighted by class support
                - recall_weighted: Recall weighted by class support
                - f1_weighted: F1 score weighted by class support

        Note:
            Returns zeros for all metrics if no valid comparisons can be made.
        """
        y_true = []
        y_pred = []

        for img in images:
            if img.cell_type_info is None or img.predicted_cell_types is None:
                continue

            common_cells = set(img.cell_type_info.keys()) & set(
                img.predicted_cell_types.keys()
            )

            y_true.extend(img.cell_type_info[idx] for idx in common_cells)
            y_pred.extend(img.predicted_cell_types[idx] for idx in common_cells)

        if not y_true:
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
            }

        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        acc = accuracy_score(y_true, y_pred)

        return {
            "accuracy": float(acc),
            "precision_macro": float(precision_m),
            "recall_macro": float(recall_m),
            "f1_macro": float(f1_m),
            "precision_weighted": float(precision_w),
            "recall_weighted": float(recall_w),
            "f1_weighted": float(f1_w),
        }

    def run_pipeline(
        self,
        images: Sequence[ImageData],
        celltype_mapping: Optional[Dict[str, str]] = None,
        channel_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[ImageData]:
        """Run the complete phenotyping pipeline.

        Args:
            images: List of ImageData objects containing raw images and metadata
            celltype_mapping: Optional mapping of cell types to standard names
            channel_mapping: Optional mapping of channels to standard names
            **kwargs: Additional model-specific arguments

        Returns:
            List of ImageData objects with updated predictions

        Note:
            Each ImageData object is validated before processing.
        """
        # Validate all images
        for image in images:
            image.validate()

        preprocessed_data = self.preprocess(images, **kwargs)
        return self.predict(preprocessed_data, images, **kwargs)


@dataclass
class DeepCellTypesConfig(PhenotyperConfig):
    """Configuration specific to DeepCell Types model.

    Extends the base PhenotyperConfig with DeepCell Types specific parameters.

    Additional Attributes:
        n_filters: Number of filters in convolutional layers
        n_heads: Number of attention heads
        n_domains: Number of domains
        embedding_dim: Dimension of embeddings
        img_feature_extractor: Type of feature extractor ('conv' or other options)
    """

    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 24
    intermediate_dir: str = "artifacts"
    n_filters: int = 256
    n_heads: int = 4
    n_domains: int = 6
    embedding_dim: int = 1024
    img_feature_extractor: str = "conv"


class PhenotyperDeepCellTypes(BasePhenotyper):
    """Implementation of BasePhenotyper for DeepCell Types model.

    This class implements the cell phenotyping pipeline using the DeepCell Types model.
    It handles all steps from data preparation through prediction while maintaining
    compatibility with the original DeepCell Types codebase.

    Example:
        ```python
        # Initialize with default settings
        phenotyper = PhenotyperDeepCellTypes()

        # Or with custom configuration
        config = DeepCellTypesConfig(
            model_path='path/to/weights.pt',
            batch_size=512,
            n_filters=128
        )
        phenotyper = PhenotyperDeepCellTypes(config)

        # Run prediction
        phenotyper.run_pipeline(
            images=[image1, image2],
            celltype_mapping={'CD3+': 'T cell'}
        )
        ```
    """

    def __init__(self, config: Optional[DeepCellTypesConfig] = None):
        """Initialize DeepCell Types phenotyper.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.dct_config = DCTConfig()
        default_config = DeepCellTypesConfig(
            model_path=str(Path(__file__).parent / "weights/model.pt")
        )
        super().__init__(config or default_config)

    def _initialize_model(self) -> CellTypeCLIPModel:
        """Initialize the DeepCell Types model with pretrained weights."""
        config = self.config

        # Load embeddings
        ct2embedding_dict = self.dct_config.get_celltype_embedding()
        marker2embedding = self.dct_config.get_channel_embedding(
            embedding_model_name="text-embedding-3-large-1024"
        )

        # Prepare embeddings
        ct_embeddings = np.zeros(
            (len(self.dct_config.ct2idx), config.embedding_dim), dtype=np.float32
        )
        marker_embeddings = np.empty_like(
            list(marker2embedding.values()), dtype=np.float32
        )

        for ct, ebd in ct2embedding_dict.items():
            if ct in self.dct_config.ct2idx:
                ct_embeddings[self.dct_config.ct2idx[ct]] = ebd

        for marker, ebd in marker2embedding.items():
            if marker in self.dct_config.marker2idx:
                marker_embeddings[self.dct_config.marker2idx[marker]] = ebd

        # Initialize model
        model = CellTypeCLIPModel(
            n_filters=config.n_filters,
            n_heads=config.n_heads,
            n_celltypes=len(self.dct_config.ct2idx),
            n_domains=config.n_domains,
            marker_embeddings=marker_embeddings,
            embedding_dim=config.embedding_dim,
            ct_embeddings=ct_embeddings,
            img_feature_extractor=config.img_feature_extractor,
        )

        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        model.to(config.device)
        model.eval()

        return model

    def preprocess(
        self,
        images: List[ImageData],
        output_path: Optional[str] = None,
        batch_size: int = 2000,
        dct_config: Optional[DCTConfig] = None,
    ) -> str:
        """Preprocess images for cell phenotyping by creating image patches.

        This method:
        1. Creates a zarr store for preprocessed data
        2. Extracts patches around each cell
        3. Normalizes patch intensities
        4. Organizes patches by cell type

        Args:
            images: List of ImageData objects to process
            output_path: Path where preprocessed data will be saved. If None,
                uses default path in intermediate_dir
            batch_size: Number of patches to process at once for memory efficiency
            dct_config: Optional custom configuration. If None, uses defaults

        Returns:
            str: Path to the created zarr store containing preprocessed patches

        Raises:
            ValueError: If no images provided or if masks are missing
            RuntimeError: If zarr store creation fails
        """
        if not images:
            raise ValueError("No images provided")

        # Initialize config if not provided
        if dct_config is None:
            dct_config = DCTConfig()

        if output_path is None:
            output_path = Path(self.config.intermediate_dir) / "preprocessed.zarr"
        else:
            output_path = Path(output_path)

        # Create output zarr store
        output_group = zarr.open(output_path, mode="w")

        # Store metadata
        output_group.attrs["channel_names"] = images[0].channel_names

        # Find all unique cell types
        unique_cell_types = set()
        for image in images:
            if image.cell_type_info:
                unique_cell_types.update(image.cell_type_info.values())
            else:
                unique_cell_types.add("Unknown")

        # Create groups for each cell type
        num_channels = len(images[0].channel_names)
        patch_shape = (
            num_channels,
            dct_config.CROP_SIZE,
            dct_config.CROP_SIZE,
        )

        for cell_type in unique_cell_types:
            ct_group = output_group.create_group(cell_type)
            ct_group.create_dataset(
                "raw",
                shape=(0, *patch_shape),
                chunks=(64, *patch_shape),
                dtype="float32",
            )
            ct_group.create_dataset(
                "mask",
                shape=(0, *(patch_shape[1:]), 2),
                chunks=(64, *(patch_shape[1:]), 2),
                dtype="float32",
            )
            ct_group.create_dataset(
                "cell_index", shape=(0,), chunks=(64,), dtype="int32"
            )
            ct_group.create_dataset(
                "file_name", shape=(0,), chunks=(64,), dtype="U100"
            )

        # Calculate quantile values for normalization
        q_values = []
        for image in images:
            raw = image.raw.astype(np.float32)
            raw[raw == 0] = np.nan
            q = np.nanquantile(raw, 0.99, axis=(1, 2))
            q_values.append(q)

        q_values = np.array(q_values)
        final_q = np.nanmean(q_values, axis=0)

        # Process each image
        for image in tqdm(images, desc="Processing images"):
            raw = image.raw.astype(np.float32)
            mask = image.predicted_mask if image.predicted_mask is not None else image.mask

            if mask is None:
                raise ValueError(f"No mask available for image {image.image_id}")

            # Get cell type information
            cell_indices = None
            cell_types = None
            if image.cell_type_info:
                cell_indices = list(image.cell_type_info.keys())
                cell_types = [image.cell_type_info[idx] for idx in cell_indices]

            # Generate and store patches
            batches = defaultdict(list)
            for raw_patch, mask_patch, idx, orig_ct in patch_generator(
                raw,
                mask,
                image.image_mpp,
                dct_config=dct_config,
                final_q=final_q,
                cell_index=cell_indices,
                cell_type=cell_types,
            ):
                batches[orig_ct].append((raw_patch, mask_patch, idx, orig_ct))

                # Process full batches
                if len(batches[orig_ct]) == batch_size:
                    self._save_batch(output_group[orig_ct], batches[orig_ct], image.image_id)
                    batches[orig_ct] = []

            # Process remaining patches
            for orig_ct, items in batches.items():
                if items:
                    self._save_batch(output_group[orig_ct], items, image.image_id)

        return str(output_path)

    def _save_batch(self, group: zarr.Group, batch: list, image_id: Union[int, str]) -> None:
        """Save a batch of patches to a zarr group, ensuring consistent shape format.

        Args:
            group: Zarr group to save to
            batch: List of tuples (raw_patch, mask_patch, cell_index, cell_type)
            image_id: Identifier for the source image
        """
        raw_patches = np.stack([x[0] for x in batch])  # Already in (N, C, H, W) format
        mask_patches = np.stack([x[1] for x in batch])  # Ensure (N, 1, H, W) format
        cell_indices = np.array([x[2] for x in batch])
        
        # Standardize mask patches if needed
        if mask_patches.ndim == 3:  # If (N, H, W)
            mask_patches = mask_patches[:, np.newaxis, ...]
        elif mask_patches.ndim == 4 and mask_patches.shape[-1] == 1:  # If (N, H, W, 1)
            mask_patches = np.transpose(mask_patches, (0, 3, 1, 2))
        
        group["raw"].append(raw_patches)
        group["mask"].append(mask_patches)
        group["cell_index"].append(cell_indices)
        group["file_name"].append(np.array([str(image_id)] * len(batch), dtype="U100"))


    def preprocess(
        self,
        images: List[ImageData],
        output_path: Optional[str] = None,
        batch_size: int = 2000,
        dct_config: Optional[DCTConfig] = None,
    ) -> str:
        """Preprocess images for cell phenotyping by creating image patches.
        
        Updates to ensure consistent shape handling.
        """
        if not images:
            raise ValueError("No images provided")
            
        # Initialize config if not provided
        if dct_config is None:
            dct_config = DCTConfig()

        if output_path is None:
            output_path = Path(self.config.intermediate_dir) / "preprocessed.zarr"
        else:
            output_path = Path(output_path)

        # Create output zarr store
        output_group = zarr.open(output_path, mode="w")

        # Store metadata
        output_group.attrs["channel_names"] = images[0].channel_names

        # Find all unique cell types
        unique_cell_types = set()
        for image in images:
            if image.cell_type_info:
                unique_cell_types.update(image.cell_type_info.values())
            else:
                unique_cell_types.add("Unknown")

        # Create groups for each cell type
        num_channels = len(images[0].channel_names)
        patch_shape = (
            num_channels,
            dct_config.CROP_SIZE,
            dct_config.CROP_SIZE,
        )

        for cell_type in unique_cell_types:
            ct_group = output_group.create_group(cell_type)
            ct_group.create_dataset(
                "raw",
                shape=(0, *patch_shape),
                chunks=(64, *patch_shape),
                dtype="float32",
            )
            ct_group.create_dataset(
                "mask",
                shape=(0, 1, patch_shape[1], patch_shape[2]),  # Ensure (N, 1, H, W) format
                chunks=(64, 1, patch_shape[1], patch_shape[2]),
                dtype="float32",
            )
            ct_group.create_dataset(
                "cell_index", shape=(0,), chunks=(64,), dtype="int32"
            )
            ct_group.create_dataset(
                "file_name", shape=(0,), chunks=(64,), dtype="U100"
            )

        # Calculate quantile values for normalization
        q_values = []
        for image in images:
            # Ensure raw is in (C, H, W) format
            raw = standardize_raw_image(image.raw.astype(np.float32))[0]
            raw[raw == 0] = np.nan
            q = np.nanquantile(raw, 0.99, axis=(1, 2))
            q_values.append(q)

        q_values = np.array(q_values)
        final_q = np.nanmean(q_values, axis=0)

        # Process each image
        for image in tqdm(images, desc="Processing images"):
            raw = standardize_raw_image(image.raw.astype(np.float32))[0]
            mask = standardize_mask(image.predicted_mask if image.predicted_mask is not None else image.mask)

            if mask is None:
                raise ValueError(f"No mask available for image {image.image_id}")

            # Get cell type information
            cell_indices = None
            cell_types = None
            if image.cell_type_info:
                cell_indices = list(image.cell_type_info.keys())
                cell_types = [image.cell_type_info[idx] for idx in cell_indices]

            # Generate and store patches
            batches = defaultdict(list)
            for raw_patch, mask_patch, idx, orig_ct in patch_generator(
                raw,
                mask[0],  # Pass 2D mask to patch generator
                image.image_mpp,
                dct_config=dct_config,
                final_q=final_q,
                cell_index=cell_indices,
                cell_type=cell_types,
            ):
                batches[orig_ct].append((raw_patch, mask_patch, idx, orig_ct))

                if len(batches[orig_ct]) == batch_size:
                    self._save_batch(output_group[orig_ct], batches[orig_ct], image.image_id)
                    batches[orig_ct] = []

            # Process remaining patches
            for orig_ct, items in batches.items():
                if items:
                    self._save_batch(output_group[orig_ct], items, image.image_id)

        return str(output_path)