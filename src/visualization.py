"""
A module for visualizing and analyzing multichannel biological images using napari.

This module provides a comprehensive visualization framework for biological image analysis,
with specialized support for:
- Multichannel fluorescence image display
- Cell segmentation mask visualization
- Cell type annotation overlays
- Side-by-side comparison of ground truth and predicted results

The primary class NapariViewer provides an intuitive interface for creating
interactive visualizations, while helper functions facilitate common visualization tasks.
"""

import napari
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy import ndimage

from .data_io import ImageData


class NapariViewer:
    """
    A viewer class for creating interactive visualizations of biological image data.

    This class provides a comprehensive interface for visualizing complex biological imaging
    data, with specialized support for cellular analysis. It handles multiple visualization
    layers including raw multichannel data, segmentation masks, and cell type annotations.

    Key features:
    - Multichannel fluorescence image display with customizable colormaps
    - Cell segmentation mask visualization with adjustable opacity
    - Cell type annotation with automatic color assignment
    - Interactive layer management (show/hide/remove)
    - Automatic legend generation for cell types

    Attributes:
        viewer (napari.Viewer): The main napari viewer instance
        cell_type_colormap (Optional[Dict[str, np.ndarray]]): Dictionary mapping cell types to RGB colors.
            None if no cell types have been visualized yet.
        _layers (Dict[str, napari.layers.Layer]): Internal mapping of layer names to napari layer objects

    Example:
        >>> image_data = ImageData(
        ...     raw=raw_image,  # Shape: (channels, H, W)
        ...     channel_names=["DAPI", "CD3", "CD20"],
        ...     mask=cell_masks,  # Shape: (1, H, W)
        ...     cell_type_info={1: "T cell", 2: "B cell"}
        ... )
        >>> viewer = NapariViewer()
        >>> viewer.show_channels(image_data)
        >>> viewer.add_cell_masks(image_data, show_labels=True)
        >>> viewer.add_legend()
    """

    def __init__(self):
        """Initialize the viewer with an empty napari window."""
        self.viewer = napari.Viewer()
        self.cell_type_colormap = None
        self._layers = {}

    def show_channels(
        self, data: ImageData, name: str = "image", colormap: str = "gray"
    ) -> None:
        """
        Display multichannel raw data as separate layers.

        Each channel is added as a separate layer with additive blending mode for
        better visualization of overlapping signals.

        Args:
            data (ImageData): ImageData object containing the raw image and channel information
            name (str, optional): Base name for the layers. Default: "image"
            colormap (str, optional): Colormap to use for displaying the channels. Default: 'gray'

        Raises:
            ValueError: If number of channels doesn't match channel names
        """
        if len(data.channel_names) != data.raw.shape[0]:
            raise ValueError("Number of channels does not match channel names")

        for idx, channel_name in enumerate(data.channel_names):
            layer = self.viewer.add_image(
                data.raw[idx],
                name=f"{name}_{channel_name}",
                colormap=colormap,
                blending="additive",
            )
            self._layers[f"{name}_{channel_name}"] = layer

    def _create_cell_type_colormap(
        self, cell_types: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Create a colormap assigning unique colors to each cell type.

        This method generates a consistent color mapping for cell types using matplotlib's
        tab20 colormap, ensuring visual distinction between different cell populations.

        Args:
            cell_types (List[str]): List of unique cell type names to map to colors

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping cell types to RGB color arrays (0-1 range)

        Raises:
            ValueError: If more than 20 cell types are provided (limitation of tab20 colormap)

        Note:
            The same cell type will always get the same color within a session, enabling
            consistent visualization across multiple images.
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

    def add_cell_masks(
        self,
        data: ImageData,
        name: str = "masks",
        opacity: float = 0.5,
        show_labels: bool = True,
        label_size: int = 8,
    ) -> None:
        """Add cell masks with optional type coloring and labels.

        Args:
            data: ImageData object containing mask and cell type information
            name: Base name for the mask layers
            opacity: Opacity of the mask layers (0-1)
            show_labels: Whether to show cell type labels
            label_size: Size of the cell type labels

        Note:
            Creates up to three layers:
            1. Binary mask layer
            2. Colored cell type layer (if cell_type_info is provided)
            3. Text labels layer (if show_labels is True)

        Raises:
            ValueError: If mask shape is invalid
        """
        if data.mask is None:
            return

        if data.mask.ndim != 3 or data.mask.shape[0] != 1:
            raise ValueError("Mask must have shape (1, height, width)")

        # Add mask layer
        mask_layer = self.viewer.add_labels(
            data.mask[0],  # Assuming shape (1, H, W)
            name=f"{name}_masks",
            opacity=opacity,
        )
        self._layers[f"{name}_masks"] = mask_layer

        # Add cell type visualization if available
        if data.cell_type_info is not None:
            # Create colormap if not exists
            if self.cell_type_colormap is None:
                self.cell_type_colormap = self._create_cell_type_colormap(
                    list(set(data.cell_type_info.values()))
                )

            # Create RGB visualization
            rgb_image = np.zeros((*data.mask[0].shape, 3))
            for cell_idx, cell_type in data.cell_type_info.items():
                if cell_idx == 0:
                    continue
                cell_color = self.cell_type_colormap[cell_type]
                rgb_image[data.mask[0] == cell_idx] = cell_color

            # Add colored mask layer
            color_layer = self.viewer.add_image(
                rgb_image,
                name=f"{name}_cell_types",
                rgb=True,
                opacity=opacity,
                blending="additive",
            )
            self._layers[f"{name}_cell_types"] = color_layer

            # Add labels if requested
            if show_labels:
                centroids = self.get_cell_centroids(data.mask[0])
                text = []
                positions = []

                for cell_idx, centroid in centroids.items():
                    if cell_idx in data.cell_type_info:
                        text.append(data.cell_type_info[cell_idx])
                        positions.append(centroid)

                text_layer = self.viewer.add_layer(
                    napari.layers.Text(
                        text=text,
                        pos=positions,
                        size=label_size,
                        color="white",
                        name=f"{name}_labels",
                        anchor="center",
                    )
                )
                self._layers[f"{name}_labels"] = text_layer

    def add_legend(self) -> None:
        """
        Add a text legend showing cell types and their corresponding colors.

        The legend is positioned in the top-left corner of the viewer with each cell type
        listed on a new line. Requires cell types to have been previously added via
        add_cell_masks.

        Note:
            Silently returns if no cell type colormap has been created yet.
            Any existing legend will be replaced.
        """
        if self.cell_type_colormap is None:
            return

        # Create legend text
        legend_text = []
        positions = []
        y = 10
        for cell_type, color in self.cell_type_colormap.items():
            legend_text.append(cell_type)
            positions.append([10, y])
            y += 20

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

    def toggle_layer_visibility(self, layer_name: str) -> None:
        """
        Toggle visibility of a specific layer.

        Args:
            layer_name (str): Name of the layer to toggle

        Note:
            Silently returns if the layer name is not found
        """
        if layer_name in self._layers:
            layer = self._layers[layer_name]
            layer.visible = not layer.visible

    def remove_layer(self, layer_name: str) -> None:
        """
        Remove a specific layer from the viewer.

        Args:
            layer_name (str): Name of the layer to remove

        Note:
            Silently returns if the layer name is not found.
            This operation permanently removes the layer and cannot be undone.
        """
        if layer_name in self._layers:
            self.viewer.layers.remove(self._layers[layer_name])
            del self._layers[layer_name]


def visualize_data(image_data: ImageData) -> NapariViewer:
    """
    Create a complete visualization setup from an ImageData object.

    This convenience function creates a NapariViewer instance and configures it with:
    1. Ground truth cell masks (if present)
    2. Predicted cell masks (if present)
    3. Cell type legend (if cell type information exists)

    Args:
        image_data (ImageData): ImageData object containing the raw image data,
            channel information, masks, and cell type annotations

    Returns:
        NapariViewer: Configured viewer instance with all available layers added

    Note:
        Raw channel visualization is disabled by default to focus on masks.
        To show channels, call show_channels() on the returned viewer.

    Example:
        >>> viewer = visualize_data(image_data)
        >>> # Show raw channels if needed
        >>> viewer.show_channels(image_data)
    """
    viewer = NapariViewer()

    # # Show raw channels
    # viewer.show_channels(image_data)

    # Add cell masks and types
    viewer.add_cell_masks(image_data, name="groundtruth", show_labels=True, opacity=0.5)

    # Add predicted masks if available
    if image_data.predicted_mask is not None:
        predicted_data = ImageData(
            raw=image_data.raw,
            channel_names=image_data.channel_names,
            mask=image_data.predicted_mask,
            cell_type_info=image_data.predicted_cell_types,
        )
        viewer.add_cell_masks(
            predicted_data, name="predicted", show_labels=True, opacity=0.3
        )

    # Add legend
    viewer.add_legend()

    return viewer
