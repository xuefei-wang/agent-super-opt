#!/usr/bin/env python3
"""
Main pipeline script for cell segmentation and visual critic evaluation.

This script demonstrates the full analysis pipeline:
1. Data loading from Npz files
2. Image preprocessing and denoising
3. Cell segmentation using either Mesmer or SAM2
4. Result visualization and evaluation
"""

from dotenv import load_dotenv

load_dotenv()

import hydra
from hydra.core.global_hydra import GlobalHydra

# Initialize Hydra config path to bypass SAM-2's settings
if not GlobalHydra().is_initialized():
    hydra.initialize(config_path="src/sam2/sam2/configs", version_base="1.2")

from pathlib import Path
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import os
from dataclasses import replace
import numpy as np
import torch
from skimage.restoration import denoise_nl_means
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd

from src.data_io import NpzDataset, ImageData
from src.segmentation import MesmerSegmenter, SAM2Segmenter

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def denoise_image(image_data: ImageData, sigma: float = 1.0) -> ImageData:
    """Apply Gaussian denoising to the image data.

    Args:
        image_data: Input ImageData object
        sigma: Standard deviation for Gaussian filter

    Returns:
        ImageData with denoised raw data
    """
    # ImageData already standardizes raw to (C, H, W) format
    raw = image_data.raw
    denoised = np.zeros_like(raw)

    # Apply denoising to each channel
    for i in range(raw.shape[0]):
        denoised[i] = gaussian_filter(raw[i], sigma=sigma)

    return replace(image_data, raw=denoised)


def set_gpu_device(gpu_id: int) -> None:
    """Set global GPU device for both PyTorch and TensorFlow."""
    if torch.cuda.is_available():
        # Set CUDA device for PyTorch
        torch.cuda.set_device(gpu_id)
        # Set environment variable for TensorFlow
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Using GPU device {gpu_id}")
    else:
        logger.warning("No GPU available, using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    """Save evaluation metrics to a text file.

    Args:
        metrics: Dictionary of metric names and values
        output_path: Directory to save the metrics file
    """
    metrics_file = output_path / "metrics.txt"
    with open(metrics_file, "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")


def run_pipeline(
    data_path: str,
    output_dir: str = "output",
    segmenter_type: str = "mesmer",
    save_visualization: bool = True,
    interactive_viz: bool = False,
    seed: int = 42,
) -> Dict[str, ImageData]:
    """Run the complete analysis pipeline.

    Args:
        data_path: Path to input dataset (.npz file)
        output_dir: Directory to save results
        segmenter_type: Type of segmenter to use ('mesmer' or 'sam2')
        save_visualization: Whether to save static visualizations
        interactive_viz: Whether to show interactive napari visualization

    Returns:
        Dictionary mapping file names to their fully processed ImageData objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set up SAM2 model config and path
    PROJECT_ROOT = Path(__file__).parent
    model_cfg = "sam2.1/sam2.1_hiera_t.yaml"  # Config path has already been set to it parent folder
    checkpoint_path = str(PROJECT_ROOT / "src/sam2/checkpoints/sam2.1_hiera_tiny.pt")

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    logger.info(f"Loading data from {data_path}")
    try:
        dataset = NpzDataset(data_path)
        # images = dataset.load_all()
        images = dataset.load(
            np.random.choice(len(dataset), 3)
        )  # Load only 3 images for testing
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # Initialize segmenter
    if segmenter_type.lower() == "mesmer":
        segmenter = MesmerSegmenter()
    elif segmenter_type.lower() == "sam2":
        segmenter = SAM2Segmenter(
            model_cfg=model_cfg,
            checkpoint_path=checkpoint_path,
        )
    else:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}")

    processed_images = {}
    metrics_list = []

    # Process each image
    for image in images:
        logger.info(f"Processing image {image.image_id}")

        # Denoise
        logger.info("Applying denoising")
        denoised_image = denoise_image(image)

        # Run segmentation
        logger.info("Running segmentation")
        try:
            segmented_image = segmenter.predict(denoised_image)
            processed_images[str(image.image_id)] = segmented_image
            if segmented_image.mask is not None:
                logger.info("Calculating evaluation metrics")
                metrics = segmenter.calculate_object_metrics(
                    segmented_image.mask, segmented_image.predicted_mask
                )
                logger.info(f"Metrics: {metrics}")
                metrics_list.append(metrics)
            else:
                logging.info(
                    "Grountruth mask not available, skipping metric calculation"
                )

        except Exception as e:
            logger.error(f"Segmentation failed for image {image.image_id}: {str(e)}")
            continue

        # Save segmentation mask
        if segmented_image.predicted_mask is not None:
            mask_path = output_dir / f"{image.image_id:03d}_predicted_mask.npy"
            np.save(mask_path, segmented_image.predicted_mask)

    # Calculate and save metrics if ground truth is available
    if any(img.mask is not None for img in images):
        df = pd.DataFrame(metrics_list)
        avg_metrics = df.mean().to_dict()
        save_metrics(avg_metrics, output_dir)
        logger.info(f"Average Metrics: {avg_metrics}")

    return processed_images


def main():
    parser = argparse.ArgumentParser(
        description="Run cell segmentation and phenotyping pipeline"
    )
    parser.add_argument("--data_path", required=True, help="Path to input npz dataset")
    parser.add_argument(
        "--output", default="output", help="Output directory for results"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument(
        "--segmenter",
        choices=["mesmer", "sam2"],
        default="mesmer",
        help="Segmentation model to use",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable static visualization saving"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive napari visualization",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    try:
        set_gpu_device(args.gpu)
        run_pipeline(
            args.data_path,
            args.output,
            segmenter_type=args.segmenter,
            save_visualization=not args.no_viz,
            interactive_viz=args.interactive,
            seed=args.seed,
        )
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
