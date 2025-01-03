import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging
from pathlib import Path

from src.data_io import NpzDataset, ImageData

def extract_image_features(image: ImageData) -> np.ndarray:
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
        1D numpy array of features
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
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.percentile(channel_data, 10),
            np.percentile(channel_data, 50),
            np.percentile(channel_data, 90)
        ])
    
    # Extract mask-based features if available
    if image.mask is not None:
        mask = image.mask[0] if image.mask.ndim == 3 else image.mask
        unique_cells = np.unique(mask)[1:]  # Exclude background
        num_cells = len(unique_cells)
        avg_cell_size = np.mean([np.sum(mask == cell_id) for cell_id in unique_cells]) if num_cells > 0 else 0
        features.extend([num_cells, avg_cell_size])
    else:
        features.extend([0, 0])  # Placeholder if no mask
        
    return np.array(features)

def select_diverse_samples(
    dataset_path: str,
    num_samples: int = 20,
    random_seed: int = 42
) -> Tuple[List[ImageData], List[int]]:
    """Select diverse representative samples from a dataset using clustering.
    
    Args:
        dataset_path: Path to .npz dataset
        num_samples: Number of diverse samples to select
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (selected ImageData objects, their indices in original dataset)
    """
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = NpzDataset(dataset_path)
    images = dataset.load_all()
    
    if len(images) < num_samples:
        raise ValueError(f"Dataset contains only {len(images)} images, "
                       f"cannot select {num_samples} samples")
    
    # Extract features for all images
    logging.info("Extracting image features")
    features = []
    for img in images:
        img_features = extract_image_features(img)
        features.append(img_features)
    
    feature_matrix = np.array(features)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Perform k-means clustering
    logging.info(f"Performing k-means clustering with k={num_samples}")
    kmeans = KMeans(
        n_clusters=num_samples,
        random_state=random_seed,
        n_init=10
    )
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
            normalized_features[cluster_points] - cluster_center,
            axis=1
        )
        closest_point_idx = cluster_points[np.argmin(distances)]
        selected_indices.append(closest_point_idx)
    
    # Get corresponding ImageData objects
    selected_images = [images[idx] for idx in selected_indices]
    
    logging.info(f"Selected {len(selected_images)} diverse samples")
    return selected_images, selected_indices

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Select diverse representative images from a dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to validation dataset (.npz file)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of diverse samples to select"
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Output directory for selected images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Select diverse samples
    selected_images, selected_indices = select_diverse_samples(
        args.dataset,
        args.num_samples,
        args.seed
    )
    
    # Stack selected images and masks
    X = np.stack([img.raw for img in selected_images])
    y = np.stack([img.mask if img.mask is not None else np.zeros_like(img.raw) 
                  for img in selected_images])

    # Save in NPZ format matching the input dataset structure
    output_path = output_dir / "diverse_samples.npz"
    np.savez_compressed(output_path, X=X, y=y)
    
    # Save indices for reference
    np.save(output_dir / "selected_indices.npy", selected_indices)
    logging.info(f"Saved {len(selected_images)} diverse samples to {output_path}")