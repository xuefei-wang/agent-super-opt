"""
A module for visualizing and analyzing multichannel biological images using both
napari (for interactive visualization) and matplotlib (for static image output).

This module provides a comprehensive visualization framework for biological image analysis,
supporting both interactive and static visualization modes with specialized features for:
- Multichannel fluorescence image display
- Cell segmentation mask visualization
- Cell type annotation overlays
- Side-by-side comparison of ground truth and predicted results
"""

import os
import napari
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from dataclasses import dataclass, replace

from .data_io import ImageData


@dataclass
class VisConfig:
    """Configuration for visualization settings."""

    output_dir: Optional[Path] = None  # Directory for saving static visualizations
    dpi: int = 300  # DPI for saved images
    figsize: tuple = (10, 10)  # Figure size for matplotlib plots
    show_raw: bool = True  # Whether to show raw channel data
    show_predicted: bool = True  # Whether to show predicted masks/types
    opacity: float = 0.5  # Opacity for mask overlays
    label_size: int = 8  # Size of cell type labels


class BaseVisualizer:
    """Base class for visualization functionality shared between interactive and static modes."""

    def _create_cell_type_colormap(
        self, cell_types: List[str]
    ) -> Dict[str, np.ndarray]:
        """Create a colormap assigning unique colors to each cell type.

        Args:
            cell_types: List of unique cell type names to map to colors

        Returns:
            Dictionary mapping cell types to RGB color arrays (0-1 range)

        Raises:
            ValueError: If more than 20 cell types provided
        """
        n_types = len(cell_types)
        if n_types > 20:
            raise ValueError("Too many cell types (maximum 20 supported)")

        cmap = plt.cm.get_cmap("tab20")
        return {
            cell_type: np.array(cmap(i)[:3]) for i, cell_type in enumerate(cell_types)
        }

    def get_cell_centroids(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate centroid coordinates for each cell in the mask.

        Args:
            mask: 2D array where each value is a cell index

        Returns:
            Dictionary mapping cell indices to their centroid coordinates (y, x)
        """
        centroids = {}
        for cell_idx in np.unique(mask):
            if cell_idx == 0:  # Skip background
                continue
            cell_mask = mask == cell_idx
            centroid = ndimage.center_of_mass(cell_mask)
            centroids[cell_idx] = np.array(centroid)
        return centroids


class NapariViewer(BaseVisualizer):
    """Interactive viewer class using napari for visualization."""

    def __init__(self, config: VisConfig):
        """Initialize the viewer with an empty napari window."""
        self.config = config
        self.viewer = napari.Viewer()
        self.cell_type_colormap = None
        self._layers = {}

    def show_channels(self, data: ImageData, name: str = "image") -> None:
        """Display multichannel raw data as separate layers."""
        if data.raw is None:
            return

        # Data should already be in (C, H, W) format from ImageData standardization
        if len(data.channel_names) != data.raw.shape[0]:
            raise ValueError("Number of channels does not match channel names")

        for idx, channel_name in enumerate(data.channel_names):
            layer = self.viewer.add_image(
                data.raw[idx],
                name=f"{name}_{channel_name}",
                colormap="gray",
                blending="additive",
            )
            self._layers[f"{name}_{channel_name}"] = layer

    def add_cell_masks(self, data: ImageData, name: str = "masks") -> None:
        """Add cell masks with optional type coloring and labels."""
        if data.mask is None:
            return

        # Get 2D mask from standardized (1, H, W) format
        mask = data.mask[0]

        # Add mask layer
        mask_layer = self.viewer.add_labels(
            mask,
            name=f"{name}_masks",
            opacity=self.config.opacity,
        )
        self._layers[f"{name}_masks"] = mask_layer

        # Add cell type visualization if available
        if data.cell_type_info is not None:
            if self.cell_type_colormap is None:
                self.cell_type_colormap = self._create_cell_type_colormap(
                    list(set(data.cell_type_info.values()))
                )

            # Create RGB visualization
            rgb_image = np.zeros((*mask.shape, 3))
            for cell_idx, cell_type in data.cell_type_info.items():
                cell_color = self.cell_type_colormap[cell_type]
                rgb_image[mask == cell_idx] = cell_color

            # Add colored mask layer
            color_layer = self.viewer.add_image(
                rgb_image,
                name=f"{name}_cell_types",
                rgb=True,
                opacity=self.config.opacity,
                blending="additive",
            )
            self._layers[f"{name}_cell_types"] = color_layer

            # Add labels
            centroids = self.get_cell_centroids(mask)
            text = []
            positions = []

            for cell_idx, centroid in centroids.items():
                if cell_idx in data.cell_type_info:
                    text.append(data.cell_type_info[cell_idx])
                    positions.append(centroid)

            if text and positions:
                text_layer = self.viewer.add_layer(
                    napari.layers.Text(
                        text=text,
                        pos=positions,
                        size=self.config.label_size,
                        color="white",
                        name=f"{name}_labels",
                        anchor="center",
                    )
                )
                self._layers[f"{name}_labels"] = text_layer

    def add_legend(self, position: str = "top-left") -> None:
        """Add a text legend showing cell types and their corresponding colors."""
        if self.cell_type_colormap is None:
            return

        # Calculate position coordinates based on viewer size
        canvas_size = self.viewer.window.qt_viewer.canvas.size()
        if position == "top-right":
            x_base = canvas_size.width() - 150
            y_base = 10
        elif position == "bottom-left":
            x_base = 10
            y_base = canvas_size.height() - 20 * len(self.cell_type_colormap) - 10
        elif position == "bottom-right":
            x_base = canvas_size.width() - 150
            y_base = canvas_size.height() - 20 * len(self.cell_type_colormap) - 10
        else:  # top-left
            x_base = 10
            y_base = 10

        # Create legend text
        legend_text = []
        positions = []
        y = y_base
        for cell_type, color in self.cell_type_colormap.items():
            legend_text.append(cell_type)
            positions.append([x_base, y])
            y += 20

        if "legend" in self._layers:
            self.viewer.layers.remove(self._layers["legend"])

        legend_layer = self.viewer.add_layer(
            napari.layers.Text(
                text=legend_text,
                pos=positions,
                size=12,
                color="white",
                name="cell_type_legend",
            )
        )
        self._layers["legend"] = legend_layer

    def clear(self) -> None:
        """Remove all layers from the viewer."""
        self.viewer.layers.clear()
        self._layers.clear()
        self.cell_type_colormap = None


class MatplotlibVisualizer(BaseVisualizer):
    """Static visualization class using matplotlib for saving image outputs."""

    def __init__(self, config: VisConfig):
        """Initialize the matplotlib visualizer with configuration."""
        self.config = config
        self.cell_type_colormap = None
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)

    def plot_channels(self, data: ImageData, prefix: str = "") -> None:
        """Plot each channel separately and save to files."""
        if data.raw is None or not data.channel_names:
            return

        # Data should already be in (C, H, W) format
        for idx, channel_name in enumerate(data.channel_names):
            plt.figure(figsize=self.config.figsize)
            plt.imshow(data.raw[idx], cmap="gray")
            plt.axis("off")
            plt.title(f"Channel: {channel_name}")

            if self.config.output_dir:
                output_path = (
                    self.config.output_dir
                    / f"{data.image_id}_channel_{channel_name}.png"
                )
                plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()

    def _prepare_mask_visualization(
        self, mask: np.ndarray, cell_type_info: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """Prepare mask for visualization, ensuring correct dimensions and coloring."""
        # Ensure mask is 2D
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]  # Handle (H, W, 1) format
            else:
                mask = mask[0]  # Handle (1, H, W) format

        if cell_type_info is not None:
            if self.cell_type_colormap is None:
                self.cell_type_colormap = self._create_cell_type_colormap(
                    list(set(cell_type_info.values()))
                )

            # Create RGB visualization
            rgb_image = np.zeros((*mask.shape, 3))
            for cell_idx, cell_type in cell_type_info.items():
                cell_color = self.cell_type_colormap[cell_type]
                rgb_image[mask == cell_idx] = cell_color
            return rgb_image
        else:
            # Create colored mask visualization without cell types
            cmap = plt.cm.get_cmap("tab20")
            rgb_image = np.zeros((*mask.shape, 3))
            unique_cells = np.unique(mask)[1:]  # Skip background
            for i, cell_idx in enumerate(unique_cells):
                rgb_image[mask == cell_idx] = cmap(i % 20)[:3]
            return rgb_image

    def plot_comparison(self, data: ImageData) -> None:
        """Plot ground truth and predicted masks side by side with combined visualization."""
        if data.mask is None:
            return

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(self.config.figsize[0] * 3, self.config.figsize[1])
        )

        # Ground truth visualization (using first channel of standardized mask)
        gt_vis = self._prepare_mask_visualization(data.mask[0], data.cell_type_info)
        ax1.imshow(gt_vis)
        ax1.set_title("Ground Truth")
        ax1.axis("off")

        # Predicted mask visualization
        if data.predicted_mask is not None:
            pred_vis = self._prepare_mask_visualization(
                data.predicted_mask[0], data.predicted_cell_types
            )
            ax2.imshow(pred_vis)
            ax2.set_title("Prediction")
        ax2.axis("off")

        # Combined visualization
        if data.predicted_mask is not None:
            # Create overlay
            combined = np.zeros((*gt_vis.shape[:2], 3))
            combined[..., 0] = data.mask[0] > 0  # Red channel for ground truth
            combined[..., 1] = (
                data.predicted_mask[0] > 0
            )  # Green channel for prediction
            ax3.imshow(combined)
            ax3.set_title("Overlay (GT=Red, Pred=Green)")
        ax3.axis("off")

        # Add legend if cell type information is available
        if self.cell_type_colormap is not None:
            legend_elements = [
                Patch(facecolor=color, label=cell_type)
                for cell_type, color in self.cell_type_colormap.items()
            ]
            fig.legend(
                handles=legend_elements, loc="center right", bbox_to_anchor=(0.98, 0.5)
            )

        plt.tight_layout()

        if self.config.output_dir:
            output_path = self.config.output_dir / f"{data.image_id:03d}_comparison.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()


def visualize(
    image_data: ImageData, mode: str = "both", config: Optional[VisConfig] = None
) -> Optional[NapariViewer]:
    """
    Unified visualization function supporting both interactive and static modes.

    Args:
        image_data: ImageData object containing all data to visualize
        mode: Visualization mode ('interactive', 'static', or 'both')
        config: Configuration for visualization settings

    Returns:
        NapariViewer instance if mode is 'interactive' or 'both', None otherwise

    Raises:
        ValueError: If mode is not one of 'interactive', 'static', or 'both'
    """
    if mode not in ["interactive", "static", "both"]:
        raise ValueError("Mode must be 'interactive', 'static', or 'both'")

    config = config or VisConfig()
    viewer = None

    # Interactive visualization with napari
    if mode in ["interactive", "both"]:
        viewer = NapariViewer(config)

        if config.show_raw and image_data.channel_names is not None:
            viewer.show_channels(image_data)

        # Add ground truth masks and types
        if image_data.mask is not None:
            viewer.add_cell_masks(image_data, name="groundtruth")

        # Add predicted masks if available and requested
        if config.show_predicted and image_data.predicted_mask is not None:
            predicted_data = replace(
                image_data,
                mask=image_data.predicted_mask,
                cell_type_info=image_data.predicted_cell_types,
            )
            viewer.add_cell_masks(predicted_data, name="predicted")

        viewer.add_legend()

    # Static visualization with matplotlib
    if mode in ["static", "both"]:
        if not config.output_dir:
            raise ValueError("output_dir must be specified for static visualization")

        visualizer = MatplotlibVisualizer(config)

        # Plot raw channels
        if config.show_raw:
            visualizer.plot_channels(image_data)

        # Plot comparison visualization
        visualizer.plot_comparison(image_data)

    return viewer
