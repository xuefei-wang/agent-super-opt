from typing import Tuple
from deepcell_spots.applications import SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.utils.augmentation_utils import subpixel_distance_transform
from tensorflow.keras.utils import to_categorical
import numpy as np

from src.data_io import ImageData
from abc import ABC, abstractmethod
from deepcell_spots.point_metrics import stats_points
from deepcell_spots.utils.postprocessing_utils import y_annotations_to_point_list_max


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

    def prepare_images(self, data_path: str) -> Tuple[ImageData, np.ndarray]:
        '''Load in data from [data_path] and return the images as an ImageData object and ground truth annotations as a numpy array.'''
        spots_data = np.load(data_path, allow_pickle=True)
        images = spots_data['X']
        spots_truth = spots_data['y']
        images = ImageData(raw=images, batch_size=images.shape[0], image_ids=[i for i in range(images.shape[0])])
        return images, spots_truth    

    def baseline(self, data_path) -> np.ndarray:
        """
        Baseline function to evaluate the model on data from [data_path] without any preprocessing or postprocessing.

        Parameters:
        - images: ImageData object containing the input images.

        Returns: 
            Dict[str, float]: Dictionary with metrics:
                - class_loss: Loss of predicted spots on one-hot encoded image matrix to ground truth 
                - regress_loss: Loss of predicted spots on regression distance image matrix to ground truth
                - precision: Fraction of predicted spots that are true positives
                - recall: Fraction of true positives that are predicted spots
                - F1: Harmonic mean of precision and recall
        """
        

        images, spots_truth = self.prepare_images(data_path)

        pred = self.predict(images)

        evals = self.evaluate(pred, spots_truth)

        return evals
    
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

        pred = app.predict(images.to_numpy().raw, batch_size=images.batch_size, threshold=0.95)

        return pred
    
    def evaluate(self, pred: np.ndarray, truth: np.ndarray) -> dict:
        """
        Evaluate metrics from detections against ground truth using the model output.
        
        Parameters:
            - pred: ndarray with two keys ['classification', 'offset_regression']
            - truth: ndarray, list of x and y coordinates of spots per image

        Returns: 
            Dict[str, float]: Dictionary with metrics:
                - class_loss: Loss of predicted spots on one-hot encoded image matrix to ground truth 
                - regress_loss: Loss of predicted spots on regression distance image matrix to ground truth
                - precision: Fraction of predicted spots that are true positives
                - recall: Fraction of true positives that are predicted spots
                - F1: Harmonic mean of precision and recall
        """

        # Source: deepcell-spots
        def get_mean_stats(y_test, y_pred, threshold=0.98, d_thresh=1):
            """Calculates the precision, recall, F1 score, and sum of min distances
            for stack of predictions.

            Args:
                y_test (array): Array of shape `(N1,d),` set of `N1` points in `d` dimensions.
                y_pred (array): A batch of predictions, of the format: `y_pred[annot_type][ind]`
                    is an annotation for image ind in the batch where annot_type = 0
                    or 1: 0 - `classification` (from classification head),
                    1 - `offset_regression` (from regression head).
                threshold (float): Probability threshold for determining spot locations.
                d_thresh (float): A distance threshold used in the definition of `tp` and `fp`.
            """
            n_test = len(y_test)  # number of test images

            precision_list = [None] * n_test
            recall_list = [None] * n_test
            F1_list = [None] * n_test

            y_pred = y_annotations_to_point_list_max(y_pred, threshold)
            for ind in range(n_test):  # loop over test images
                s = stats_points(y_test[ind], y_pred[ind], threshold=d_thresh)
                precision_list[ind] = s['precision']
                recall_list[ind] = s['recall']
                F1_list[ind] = s['F1']

            precision = np.mean(precision_list)
            recall = np.mean(recall_list)
            F1 = np.mean(F1_list)

            return {"precision": precision, "recall": recall, "F1": F1}
        
        # Source: deepcell-spots
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
        

        stats_metrics = get_mean_stats(truth, pred)
        
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
        
        eval_metrics = {
            "class_loss": (batch_class_loss / truth.shape[0]),
            "regress_loss": (batch_regress_loss / truth.shape[0])
        }

        eval_metrics.update(stats_metrics)
        return eval_metrics