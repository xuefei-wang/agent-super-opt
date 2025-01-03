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
    
    # Load diverse samples
    logging.info(f"Loading samples from {npz_path}")
    dataset = NpzDataset(npz_path)
    images = dataset.load_all()
    logging.info(f"Loaded {len(images)} images")
    
    # Create visualizations for each image
    for i, img in enumerate(images):
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Sample {i}' + (f' (Original Index: {selected_indices[i]})' if selected_indices is not None else ''))
        
        # Plot raw image
        raw = img.raw
        if raw.ndim == 3:
            if raw.shape[0] == 1:  # (1, H, W)
                raw = raw[0]
            elif raw.shape[-1] == 1:  # (H, W, 1)
                raw = raw[..., 0]
        im1 = ax1.imshow(raw, cmap='gray')
        ax1.set_title('Raw Image')
        ax1.axis('off')
        # plt.colorbar(im1, ax=ax1)
        
        # Plot mask
        mask = img.mask
        if mask is not None:
            if mask.ndim == 3:
                if mask.shape[0] == 1:  # (1, H, W)
                    mask = mask[0]
                elif mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask[..., 0]
            
            # Create colored mask visualization
            from scipy import ndimage
            unique_cells = np.unique(mask)
            colored_mask = np.zeros((*mask.shape, 3))
            
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
        orig_id = selected_indices[i]
        output_path = output_dir / f"diverse_sample_{i:02d}_{orig_id:03d}.png"
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