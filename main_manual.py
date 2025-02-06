#!/usr/bin/env python3
"""
Simplified main pipeline script for cell segmentation.
"""

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
import pandas as pd

from src.data_io import NpzDataset, ZarrDataset
from src.segmentation import MesmerSegmenter, SAM2Segmenter, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def run_pipeline(
    data_path: str,
    output_dir: str = "output",
    segmenter_type: str = "mesmer",
    gpu_id: int = 0,
    seed: int = 42
) -> None:
    """
    Simple pipeline for cell segmentation.
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set GPU device
    set_gpu_device(gpu_id)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    # Load dataset
    try:
        dataset = NpzDataset(data_path) if data_path.endswith('.npz') else ZarrDataset(data_path)
        indices = np.random.choice(len(dataset), size=5, replace=False)
        image_data = dataset.load(indices)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Initialize and run segmenter
    try:
        if segmenter_type.lower() == "mesmer":
            segmenter = MesmerSegmenter()
        else:  # sam2
            PROJECT_ROOT = Path(__file__).parent
            segmenter = SAM2Segmenter(
                model_cfg="sam2.1/sam2.1_hiera_t.yaml",
                checkpoint_path=str(PROJECT_ROOT / "src/sam2/checkpoints/sam2.1_hiera_tiny.pt")
            )
            
        # Run segmentation on entire dataset at once
        results = segmenter.predict(image_data)

        # Calculate and save metrics if ground truth is available
        if results.masks is not None:
            metrics = calculate_metrics(results.masks, results.predicted_masks)
            df = pd.DataFrame(metrics)
            df['image_id'] = results.image_ids
            df.to_csv(output_dir / "metrics.csv", index=False)
            
            # Log overall metrics
            overall_metrics = df.drop('image_id', axis=1).mean().to_dict()
            logger.info(f"Overall metrics: {overall_metrics}")
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise

    # Save results
    try:
        # Save predicted masks
        for i, image_id in enumerate(results.image_ids):
            mask_path = output_dir / f"{image_id:03d}_predicted_mask.npy"
            np.save(mask_path, results.predicted_masks[i])
            
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run cell segmentation pipeline")
    parser.add_argument("--data_path", required=True, help="Path to input dataset (.npz or Zarr)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--segmenter", choices=["mesmer", "sam2"], default="mesmer", help="Segmentation model type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        run_pipeline(
            data_path=args.data_path,
            output_dir=args.output,
            segmenter_type=args.segmenter,
            gpu_id=args.gpu,
            seed=args.seed
        )
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()