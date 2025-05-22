
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData



class BaseSegmenter(ABC):
    """Abstract base class defining the interface for cell segmentation models.

    This class provides a standardized interface that all segmentation model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction while remaining model-agnostic.

    All implementations must handle batched inputs and outputs consistently,
    following the shape conventions defined in the ImageData class.

    Methods:
        preprocess: Prepare raw image data for model input
        predict: Generate segmentation predictions for the input images

    Example Implementation:
        >>> class MySegmenter(BaseSegmenter):
        ...     def preprocess(self, image_data):
        ...         # Implementation
        ...         pass
        ...
        ...     def predict(self, image_data, **kwargs):
        ...         # Implementation
        ...         pass

    Notes:
        - All methods must preserve batch dimensions
        - Implementations should handle both single-channel and multi-channel inputs
        - Error handling should be comprehensive and informative
        - Memory efficiency should be considered for large batches
    """
    @abstractmethod
    def preprocess(
        self,
        image_data: ImageData,
    ) -> np.ndarray:
        """Prepare images for model input.
        
        Returns preprocessed numpy array regardless of underlying framework.
        
        Args:
            image_data: Input images and metadata
            
        Returns:
            Preprocessed array in model's expected format
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_data: ImageData,
        **kwargs
    ) -> ImageData:
        """Generate segmentation predictions.
        
        Takes framework-agnostic input and returns framework-agnostic output.
        
        Args:
            image_data: Input images and metadata
            **kwargs: Additional model parameters
            
        Returns:
            ImageData with predictions populated
        """
        pass

class BaseSpotDetector(ABC):
    """Abstract base class defining the interface for cell spot detection models.

    This class provides a standardized interface that all spot detection model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction while remaining model-agnostic.

    All implementations must handle batched inputs and outputs consistently,
    following the shape conventions defined in the ImageData class.

    Methods:
        predict: Generate cell spot predictions for the input images
        evaluate: Generate loss values and metrics to compare predictions to ground truth

    Example Implementation:
        >>> class MySpotDetector(BaseSpotDetector):
        ...     def preprocess(self, image_data):
        ...         # Implementation
        ...         pass
        ...
        ...     def predict(self, image_data):
        ...         # Implementation
        ...         pass

    Notes:
        - All methods must preserve batch dimensions
        - Implementations should handle both single-channel and multi-channel inputs
        - Error handling should be comprehensive and informative
        - Memory efficiency should be considered for large batches
    """

    @abstractmethod
    def predict(
        self,
        image_data: ImageData,
        **kwargs
    ) -> np.ndarray:
        """Generate spot detection predictions.
        
        Takes framework-agnostic input and returns framework-agnostic output.
        
        Args:
            image_data: Input images and metadata
            **kwargs: Additional model parameters
            
        Returns:
            Model output as an ndarray
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate loss and metrics of predicted spot locations to the ground truth.

        This function computes various evaluation metrics to assess the quality of cell
        detection predictions against ground truth spot locations. It handles batched inputs
        and can return either per-batch metrics or averaged metrics across the batch.
        

        Args:
            pred: Predicted spot images in an ndarray. See specific model implementation for data structure.
            truth: Ground truth list of x and y coordinate locations for spots in each image with shape (B, N, 2).

        Returns:
            List[Dict[str, float]]: List of dictionaries (one per batch item) with metrics:
                - class_loss: Mean IoU of correctly matched objects
                - regress_loss: Fraction of predicted objects that match ground truth
            """
        pass