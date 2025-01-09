"""
Tool for selecting diverse samples from biological image datasets.

This module provides functionality for selecting representative samples from a large 
image dataset for validation and visual inspection. It uses feature extraction and 
k-means clustering to identify diverse examples that represent different image 
characteristics in the dataset.

Key Features:
- Automatic feature extraction from multichannel images and masks
- K-means clustering for diverse sample selection
- Support for both single and multichannel images
- Built-in handling of ImageData objects

Example:
    >>> from src.select_visual_sample import select_diverse_samples
    >>> # Select 20 diverse samples
    >>> samples, indices = select_diverse_samples(
    ...     dataset_path='path/to/dataset.npz',
    ...     num_samples=20
    ... )
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging
from pathlib import Path

from .data_io import NpzDataset, ImageData


def _extract_image_features(image: ImageData) -> np.ndarray:
    """Extract features from an image for diversity comparison.

    Features include:
    - Mean intensity per channel
    - Standard deviation per channel
    - Percentiles (10, 50, 90) per channel
    - Number of cells (if mask available)
    - Average cell size (if mask available)

    Args:
        image: ImageData object containing raw image and optional mask

    Returns:
        np.ndarray: 1D array of features with shape (n_features,), where n_features
            depends on number of channels (5 features per channel + 2 mask features)

    Notes:
        - For multichannel images, features are extracted per channel
        - If no mask is available, cell-based features are set to 0
        - Input images are automatically standardized to (C, H, W) format
    """
    features = []
    raw = image.raw

    # Handle different input formats
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]  # Add channel dimension
    elif raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw.transpose(2, 0, 1)  # Convert to (C, H, W)

    # Extract intensity-based features for each channel
    for channel in range(raw.shape[0]):
        channel_data = raw[channel]
        features.extend(
            [
                np.mean(channel_data),
                np.std(channel_data),
                np.percentile(channel_data, 10),
                np.percentile(channel_data, 50),
                np.percentile(channel_data, 90),
            ]
        )

    # Extract mask-based features if available
    if image.mask is not None:
        mask = image.mask[0] if image.mask.ndim == 3 else image.mask
        unique_cells = np.unique(mask)[1:]  # Exclude background
        num_cells = len(unique_cells)
        avg_cell_size = (
            np.mean([np.sum(mask == cell_id) for cell_id in unique_cells])
            if num_cells > 0
            else 0
        )
        features.extend([num_cells, avg_cell_size])
    else:
        features.extend([0, 0])  # Placeholder if no mask

    return np.array(features)


def select_diverse_samples(
    dataset_path: str, num_samples: int = 20, random_seed: int = 42
) -> Tuple[List[ImageData], List[int]]:
    """Select diverse representative samples from a dataset using clustering.

    Args:
        dataset_path: Path to .npz dataset containing images and masks
        num_samples: Number of diverse samples to select. Should be less than
            total dataset size and appropriate for dataset diversity
        random_seed: Random seed for k-means clustering reproducibility

    Returns:
        Tuple containing:
            - List[ImageData]: Selected diverse image samples
            - List[int]: Indices of selected samples in original dataset

    Raises:
        ValueError: If num_samples is greater than dataset size
        FileNotFoundError: If dataset_path does not exist
    """
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = NpzDataset(dataset_path)
    images = dataset.load_all()

    if len(images) < num_samples:
        raise ValueError(
            f"Dataset contains only {len(images)} images, "
            f"cannot select {num_samples} samples"
        )

    # Extract features for all images
    logging.info("Extracting image features")
    features = []
    for img in images:
        img_features = _extract_image_features(img)
        features.append(img_features)

    feature_matrix = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)

    # Perform k-means clustering
    logging.info(f"Performing k-means clustering with k={num_samples}")
    kmeans = KMeans(n_clusters=num_samples, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_features)

    # Select samples closest to cluster centers
    selected_indices = []
    for i in range(num_samples):
        cluster_points = np.where(cluster_labels == i)[0]
        if len(cluster_points) == 0:
            continue

        # Find point closest to cluster center
        cluster_center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(
            normalized_features[cluster_points] - cluster_center, axis=1
        )
        closest_point_idx = cluster_points[np.argmin(distances)]
        selected_indices.append(closest_point_idx)

    # Get corresponding ImageData objects
    selected_images = [images[idx] for idx in selected_indices]

    logging.info(f"Selected {len(selected_images)} diverse samples")
    return selected_images, selected_indices