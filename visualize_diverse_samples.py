from pathlib import Path
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.data_io import NpzDataset

def visualize_diverse_samples(
    npz_path: str,
    output_dir: str = "visualizations",
    indices_path: str = None
) -> None:
    """Load diverse samples and create side-by-side visualizations of raw and mask.
    
    Args:
        npz_path: Path to diverse_samples.npz file
        output_dir: Directory to save visualizations
        indices_path: Optional path to selected_indices.npy
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load selected indices if available
    selected_indices = None
    if indices_path and Path(indices_path).exists():
        selected_indices = np.load(indices_path)
        logging.info(f"Loaded indices: {selected_indices}")
    
    # Load diverse samples using NpzDataset which handles standardization
    logging.info(f"Loading samples from {npz_path}")
    dataset = NpzDataset(npz_path)
    images = dataset.load_all()  # Will return ImageData objects with standardized formats
    logging.info(f"Loaded {len(images)} images")
    
    # Create visualizations for each image
    for i, img in enumerate(images):
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Add title with original index if available
        fig.suptitle(
            f'Sample {i}' + 
            (f' (Original Index: {selected_indices[i]})' if selected_indices is not None else '')
        )
        
        # Plot raw image
        # ImageData standardizes raw to (C, H, W) format, so take first channel
        raw = img.raw[0]  # Get first channel
        im1 = ax1.imshow(raw, cmap='gray')
        ax1.set_title('Raw Image')
        ax1.axis('off')
        
        # Plot mask
        # ImageData standardizes mask to (1, H, W) format
        if img.mask is not None:
            mask = img.mask[0]  # Get first channel
            
            # Create colored mask visualization
            from scipy import ndimage
            unique_cells = np.unique(mask)
            colored_mask = np.zeros((*mask.shape, 3))
            
            # Color each cell
            for j, cell_id in enumerate(unique_cells[1:], 1):  # Skip 0 (background)
                cell_mask = mask == cell_id
                # Calculate center of mass for label
                center = ndimage.center_of_mass(cell_mask)
                # Add colored cell to visualization
                colored_mask[cell_mask] = plt.cm.tab20(j % 20)[:3]
            
            im2 = ax2.imshow(colored_mask)
            ax2.set_title(f'Mask (Cells: {len(unique_cells)-1})')
        else:
            ax2.text(0.5, 0.5, 'No Mask Available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            ax2.set_title('Mask')
        ax2.axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Use original index in filename if available
        orig_id = selected_indices[i] if selected_indices is not None else i
        output_path = output_dir / f"diverse_sample_{i:02d}_orig_{orig_id:03d}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created visualization for sample {i}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize diverse samples")
    parser.add_argument(
        "--npz_path",
        required=True,
        help="Path to diverse_samples.npz file"
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--indices_path",
        default=None,
        help="Optional path to selected_indices.npy"
    )
    
    args = parser.parse_args()
    
    visualize_diverse_samples(
        args.npz_path,
        args.output_dir,
        args.indices_path
    )