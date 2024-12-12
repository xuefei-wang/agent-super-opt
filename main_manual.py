#!/usr/bin/env python3
"""
Main pipeline script for cell segmentation and phenotyping.

This script demonstrates the full analysis pipeline:
1. Data loading from zarr
2. Image preprocessing and denoising
3. Channel selection for segmentation
4. Cell segmentation
5. Cell phenotyping
6. Result visualization

Example usage:
    python main.py path/to/dataset.zarr
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import logging
from pathlib import Path
from typing import Dict
import os

import numpy as np
import torch
from skimage.restoration import denoise_nl_means

from data_io import ZarrDataset, ImageData
from src.segmentation import MesmerSegmenter, ChannelSpec
from src.phenotyping import PhenotyperDeepCellTypes, DeepCellTypesConfig
from visualization import visualize_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def denoise_image(image_data: ImageData) -> ImageData:
    """Apply non-local means denoising to each channel."""
    denoised = np.zeros_like(image_data.raw)
    for i in range(image_data.raw.shape[0]):
        denoised[i] = denoise_nl_means(
            image_data.raw[i], patch_size=9, patch_distance=15, h=0.1, fast_mode=True
        )
    return ImageData(
        raw=denoised,
        channel_names=image_data.channel_names,
        tissue_type=image_data.tissue_type,
        cell_types=image_data.cell_types,
        image_mpp=image_data.image_mpp,
        file_name=image_data.file_name,
        mask=image_data.mask,
        cell_type_info=image_data.cell_type_info,
    )


def select_channels(image_data: ImageData) -> ChannelSpec:
    """Select optimal channels for segmentation based on tissue type."""

    nuclear = "HH3"
    membrane = ["CD45", "CD56", "Vimentin"]

    return ChannelSpec(nuclear=nuclear, membrane=membrane)


def set_gpu_device(gpu_id: int) -> None:
    """Set global GPU device for both PyTorch and TensorFlow."""
    # Set CUDA visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Tensorflow (used by Mesmer) specific
    os.environ["TF_CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set PyTorch default device
    torch.cuda.set_device(gpu_id)


def run_pipeline(zarr_path: str, output_dir: str = "output") -> Dict[str, ImageData]:
    """Run the complete analysis pipeline.

    Args:
        zarr_path: Path to input zarr dataset
        output_dir: Directory to save results

    Returns:
        Dictionary mapping file names to their fully processed ImageData objects,
        containing raw data, segmentation masks, and cell type predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize components
    raw_dataset = ZarrDataset(zarr_path)
    # raw_images = raw_dataset.load_all()
    raw_images = raw_dataset.load(
        [
            "HBM555.ZNTK.962-1d988a3a09dd6ab19d70f6c33974eef2",
            "HBM786.VLVN.435-2d6d2131e94f60aff982c3d52a9fcb27",
        ]
    )

    segmenter = MesmerSegmenter()

    phenotyper_config = DeepCellTypesConfig(
        model_path="pretrained/model_c_patch2_skip_Greenbaum_Uterus_0.pt", batch_size=32
    )
    phenotyper = PhenotyperDeepCellTypes(phenotyper_config)

    segmented_data_list = []
    # Process each image
    for image_data in raw_images[:3]:
        logger.info(f"Processing {image_data.file_name}")

        # Denoise
        logger.info("Applying denoising")
        denoised_data = denoise_image(image_data)

        # Select channels for segmentation
        logger.info("Selecting optimal channels")
        channel_spec = select_channels(denoised_data)
        logger.info(
            f"Using nuclear: {channel_spec.nuclear}, "
            f"membrane: {', '.join(channel_spec.membrane)}"
        )

        # Run segmentation
        logger.info("Running cell segmentation")
        segmented_data = segmenter.predict(
            denoised_data, channel_spec, image_mpp=denoised_data.image_mpp
        )

        segmented_data_list.append(segmented_data)

    # Run phenotyping
    logger.info("Running cell phenotyping")
    phenotyped_data_list = phenotyper.run_pipeline(
        images=segmented_data_list,
    )

    # Calculate metrics
    logger.info("Calculating metrics")
    metrics = phenotyper.calculate_metrics(phenotyped_data_list)
    logger.info(f"Metrics: {metrics}")

    # Save results to a zarr file
    logger.info("Saving results")
    processed_dataset = ZarrDataset.create(
        "output/result.zarr", raw_dataset.get_channel_names()
    )
    processed_dataset.save_all(phenotyped_data_list)

    # Visualize the first image
    logger.info("Generating visualization")
    viewer = visualize_data(phenotyped_data_list[0])

    # Keep viewer window open (comment out for batch processing)
    viewer.viewer.app.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run cell segmentation and phenotyping pipeline"
    )
    parser.add_argument("--zarr_path", help="Path to input zarr dataset")
    parser.add_argument(
        "--output", default="output", help="Output directory for results"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")

    args = parser.parse_args()
    set_gpu_device(args.gpu)
    run_pipeline(args.zarr_path, args.output)


if __name__ == "__main__":
    main()
