from deepcell_spots.applications import SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.utils.augmentation_utils import subpixel_distance_transform
from tensorflow.keras.utils import to_categorical
import numpy as np

from src.data_io import ImageData

class DeepcellSpotsDetector:
    def __init__(self, model_name='deepcell_spots', **kwargs):
        self.kwargs = kwargs
    
    def predict(self, images: ImageData) -> np.ndarray:
        """
        Detect spots in the given image using the loaded model.

        Parameters:
        - image: numpy array of shape (H, W, C) representing the input image.

        Returns:
        - detections: numpy array of detected spots.
    """
        app = SpotDetection()

        pred = app.predict(images.raw, batch_size=images.batch_size, threshold=0.95)

        return pred
    
    def evaluate(self, image_shape, predictions, ground_truth):
        """
        Evaluate detections against ground truth

        Returns: classification loss, regression loss dictionary
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

        for i in range(predictions.shape[0]):
            annotated_pred = point_list_to_annotations(predictions[i], image_shape=image_shape)
            annotated_truth = point_list_to_annotations(ground_truth[i], image_shape=image_shape)

            class_pred = annotated_pred['detections']
            regress_pred = annotated_pred['offset']

            class_truth = annotated_truth['detections']
            regress_truth = annotated_truth['offset']

            class_loss = losses.classification_loss(class_truth, class_pred).numpy()
            regress_loss = losses.regression_loss(regress_truth, regress_pred).numpy()

            batch_class_loss += class_loss
            batch_regress_loss += regress_loss
        
        return {
            "class_loss": (batch_class_loss / predictions.shape[0]),
            "regress_loss": (batch_regress_loss / predictions.shape[0])
        }