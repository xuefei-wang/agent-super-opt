from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import replace

from deepcell.applications import Mesmer

from .data_io import ImageData


class ChannelSpec:
    """
    Specification for channel selection in cell segmentation models.

    This class defines which imaging channels should be used for segmentation,
    specifically mapping the nuclear stain and membrane/cytoplasm markers required
    by segmentation algorithms like Mesmer. Each channel name must exactly match
    a channel name in the ImageData object.

    Attributes:
        nuclear (str): Name of the nuclear staining channel (e.g., "DAPI", "Hoechst")
        membrane (List[str]): List of channels to combine for membrane/cytoplasm signal.
                            Multiple channels will be summed to create a composite signal.

    Example:
        >>> spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
    """

    def __init__(self, nuclear: str, membrane: List[str]):
        """Initialize channel specification.

        Args:
            nuclear: Name of nuclear channel (must match a name in ImageData.channel_names)
            membrane: List of channel names to combine for membrane signal
                     (each must match a name in ImageData.channel_names)
        """
        self.nuclear = nuclear
        self.membrane = membrane


class BaseSegmenter(ABC):
    """
    Abstract base class defining the interface for cell segmentation models.

    This class provides a standardized interface that all segmentation model
    implementations must follow. It defines the core methods needed for channel
    preprocessing and cell segmentation prediction.

    The interface is designed to be model-agnostic while ensuring consistent
    handling of input data and segmentation results across different implementations.

    Example:
        class MySegmenter(BaseSegmenter):
            def preprocess_channels(self, image_data, channel_spec):
                # Implementation
                pass

            def predict(self, image_data, channel_spec, **kwargs):
                # Implementation
                pass
    """

    @abstractmethod
    def preprocess_channels(
        self, image_data: ImageData, channel_spec: ChannelSpec
    ) -> np.ndarray:
        """Prepare channels according to model requirements.

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Specification of required channels

        Returns:
            np.ndarray: Processed image ready for model input

        Raises:
            ValueError: If specified channels are not found in image_data.channel_names
        """
        pass

    @abstractmethod
    def predict(
        self, image_data: ImageData, channel_spec: ChannelSpec, **kwargs
    ) -> ImageData:
        """Perform segmentation on input image.

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Specification of required channels
            **kwargs: Additional model-specific parameters

        Returns:
            ImageData: Updated ImageData object with predicted_mask field populated.
                      The predicted_mask will have shape (1, height, width) with
                      integer labels for each segmented cell.

        Raises:
            ValueError: If specified channels are not found in image_data.channel_names
            RuntimeError: If segmentation fails
        """
        pass


class MesmerSegmenter(BaseSegmenter):
    """
    Cell segmentation implementation using the Mesmer deep learning model.

    This class provides whole-cell segmentation capabilities using the Mesmer model,
    which uses both nuclear and membrane/cytoplasm channels to accurately identify
    cell boundaries. The implementation handles all necessary preprocessing and
    ensures proper formatting of inputs/outputs.

    Key Features:
    - Automatic channel preprocessing and normalization
    - Support for multiple membrane markers
    - Built-in error handling for missing channels
    - Configurable model parameters

    Example:
        >>> # Define channels to use
        >>> channel_spec = ChannelSpec(
        ...     nuclear="DAPI",
        ...     membrane=["CD44", "Na/K-ATPase"]
        ... )
        >>>
        >>> # Initialize and run segmentation
        >>> segmenter = MesmerSegmenter()
        >>> segmented_data = segmenter.predict(
        ...     image_data=image_data,
        ...     channel_spec=channel_spec
        ... )
    """

    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize Mesmer model.

        Args:
            model_kwargs: Optional dictionary of arguments passed to Mesmer initialization.
                        See Mesmer documentation for available parameters.

        Raises:
            RuntimeError: If Mesmer model initialization fails
        """
        self.model = Mesmer(**(model_kwargs or {}))

    def preprocess_channels(
        self, image_data: ImageData, channel_spec: ChannelSpec
    ) -> np.ndarray:
        """Combine appropriate channels for Mesmer input.

        This method extracts the nuclear channel and combines specified membrane
        channels into a single membrane signal. Both channels are normalized
        independently to the range [0, 1].

        Args:
            image_data: ImageData object containing raw image and metadata
            channel_spec: Specification of required channels

        Returns:
            np.ndarray: Processed image with shape (2, height, width) containing
                       [nuclear_channel, combined_membrane_channels]

        Raises:
            ValueError: If any specified channel is not found in image_data.channel_names
        """
        # Get nuclear channel
        try:
            nuc_idx = image_data.channel_names.index(channel_spec.nuclear)
            nuclear_img = image_data.raw[nuc_idx]
        except ValueError:
            raise ValueError(
                f"Nuclear channel '{channel_spec.nuclear}' not found in provided channels"
            )

        # Combine membrane channels
        membrane_indices = []
        for channel in channel_spec.membrane:
            try:
                idx = image_data.channel_names.index(channel)
                membrane_indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Membrane channel '{channel}' not found in provided channels"
                )

        membrane_img = np.zeros_like(nuclear_img)
        for idx in membrane_indices:
            membrane_img += image_data.raw[idx]

        # Normalize each channel independently
        nuclear_img = self._normalize(nuclear_img)
        membrane_img = self._normalize(membrane_img)

        return np.stack([nuclear_img, membrane_img])

    def predict(
        self, image_data: ImageData, channel_spec: ChannelSpec, **kwargs
    ) -> ImageData:
        """Segment cells using Mesmer.

        This method performs whole-cell segmentation using the Mesmer model.
        It automatically preprocesses the channels and returns an updated
        ImageData object with the segmentation results.

        Args:
            image_data: ImageData object containing raw image and metadata.
                       The raw image should have shape (channels, height, width)
            channel_spec: Specification of nuclear and membrane channels to use
            **kwargs: Additional arguments passed to Mesmer's predict method.
                     See Mesmer documentation for available parameters.

        Returns:
            ImageData: Updated copy of input ImageData with predicted_mask field
                      populated. The mask has shape (1, height, width) where each
                      integer represents a unique cell.

        Raises:
            ValueError: If required channels are not found
            RuntimeError: If segmentation fails
        """
        # Preprocess channels
        processed_img = self.preprocess_channels(image_data, channel_spec)

        # Convert to Mesmer's expected format (batch, height, width, channels)
        model_input = np.moveaxis(processed_img, 0, -1)
        model_input = model_input[np.newaxis, ...]  # Add batch dimension

        # Run prediction
        try:
            labels = self.model.predict(model_input, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Mesmer segmentation failed: {str(e)}") from e

        # Create new ImageData with predicted mask
        return replace(
            image_data,
            predicted_mask=np.squeeze(labels),
        )

    @staticmethod
    def _normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize image to [0, 1] range with handling for edge cases.

        This is an internal helper method used during channel preprocessing.

        Args:
            img (np.ndarray): Input image to normalize
            eps (float): Small epsilon value to prevent division by zero

        Returns:
            np.ndarray: Normalized image in [0, 1] range
        """
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()

        if abs(img_max - img_min) < eps:
            return np.zeros_like(img)

        return (img - img_min) / (img_max - img_min)
