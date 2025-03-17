from typing import Optional
from deepcell_spots.applications import SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.utils.augmentation_utils import subpixel_distance_transform
from tensorflow.keras.utils import to_categorical
import numpy as np

from src.data_io import ImageData
from abc import ABC, abstractmethod

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


class DeepcellSpotsDetector(BaseSpotDetector):
    """Abstract base class defining the interface for cell spot detection models.

    This class provides a standardized interface that all spot detection model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction while remaining model-agnostic.

    All implementations must handle batched inputs and outputs consistently,
    following the shape conventions defined in the ImageData class.

    Methods:
        preprocess: Prepare raw image data for model input
        predict: Generate cell spot predictions for the input images
        evaluate: 

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
    
    def __init__(self, model_name='deepcell_spots', **kwargs):
        self.kwargs = kwargs
    
    def predict(self, images: ImageData) -> np.ndarray:
        """
        Detect spots in the given image using the loaded model.

        Parameters:
        - image: numpy array of shape (H, W, C) representing the input image.

        Returns:
        Predictions with two keys ('classification', 'offset_regression')
        - classification prediction: image array of one-hot encoded classifications
        - regression prediction: image array of regression distance from predicted points
    """
        app = SpotDetection()
        
        # Disable default preprocessing and postprocessing
        app.preprocessing_fn = None
        app.postprocessing_fn = None

        pred = app.predict(images.raw, batch_size=images.batch_size, threshold=0.95)

        return pred
    
    def evaluate(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Evaluate detections against ground truth using the model output.
        
        Parameters:
            - pred: ndarray with two keys ['classification', 'offset_regression']
            - truth: ndarray, list of x and y coordinates of spots per image

        Returns: 
            List[Dict[str, float]]: List of dictionaries (one per batch item) with metrics:
                - class_loss: Mean IoU of correctly matched objects
                - regress_loss: Fraction of predicted objects that match ground truth
        """

        def point_list_to_annotations(points, image_shape, dy=1, dx=1):
            """ Generate label images used in loss calculation from point labels.

            Args:
                points (np.array): array of size (N, 2) which contains points in the format [y, x].
                image_shape (tuple): shape of 2-dimensional image.
                dy: pixel y width.
                dx: pixel x width.

            Returns:
                annotations (dict): Dictionary with two keys, `detections` and `offset`.
                    - `detections` is array of shape (image_shape,2) with pixels one hot encoding
                    spot locations.
                    - `offset` is array of shape (image_shape,2) with pixel values equal to
                    signed distance to nearest spot in x- and y-directions.
            """

            contains_point = np.zeros(image_shape)
            for ind, [y, x] in enumerate(points):
                nearest_pixel_x_ind = int(round(x / dx))
                nearest_pixel_y_ind = int(round(y / dy))
                contains_point[nearest_pixel_y_ind, nearest_pixel_x_ind] = 1

            delta_y, delta_x, _ = subpixel_distance_transform(
                points, image_shape, dy=1, dx=1)
            offset = np.stack((delta_y, delta_x), axis=-1)

            one_hot_encoded_cp = to_categorical(contains_point)

            annotations = {'detections': one_hot_encoded_cp, 'offset': offset}
            return annotations
        
        batch_class_loss = 0
        batch_regress_loss = 0

        losses = DotNetLosses()
        
        class_preds = pred['classification']
        regress_preds = pred['offset_regression']
        
        for i in range(truth.shape[0]):
            annotated_truth = point_list_to_annotations(truth[i], image_shape=class_preds.shape[1:3])

            class_pred = class_preds[i]
            regress_pred = regress_preds[i]

            class_truth = annotated_truth['detections']
            regress_truth = annotated_truth['offset']

            class_loss = losses.classification_loss(class_truth, class_pred).numpy()
            regress_loss = losses.regression_loss(regress_truth, regress_pred).numpy()

            batch_class_loss += class_loss
            batch_regress_loss += regress_loss
        
        return {
            "class_loss": (batch_class_loss / truth.shape[0]),
            "regress_loss": (batch_regress_loss / truth.shape[0])
        }