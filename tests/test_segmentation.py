import unittest
import numpy as np
from src.segmentation import calculate_object_metrics, BaseSegmenter
from src.data_io import ImageData

class TestSegmentationMetrics(unittest.TestCase):
    def setUp(self):
        # Create some basic test masks
        self.perfect_match = {
            'true': np.array([[1, 1, 0, 2, 2],
                            [1, 1, 0, 2, 2],
                            [0, 0, 0, 0, 0],
                            [3, 3, 0, 4, 4],
                            [3, 3, 0, 4, 4]]),
            'pred': np.array([[1, 1, 0, 2, 2],
                            [1, 1, 0, 2, 2],
                            [0, 0, 0, 0, 0],
                            [3, 3, 0, 4, 4],
                            [3, 3, 0, 4, 4]])
        }

        self.partial_match = {
            'true': np.array([[1, 1, 0, 2, 2],
                            [1, 1, 0, 2, 2],
                            [0, 0, 0, 0, 0],
                            [3, 3, 0, 4, 4],
                            [3, 3, 0, 4, 4]]),
            'pred': np.array([[1, 1, 0, 2, 2],
                            [1, 0, 0, 2, 2],
                            [0, 0, 0, 0, 0],
                            [3, 3, 0, 4, 0],
                            [3, 3, 0, 4, 0]])
        }

        self.no_match = {
            'true': np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 2, 2, 2]]),
            'pred': np.array([[0, 0, 3, 3, 0],
                            [0, 0, 3, 3, 0],
                            [4, 4, 0, 0, 0]]),
        }

        self.empty_masks = {
            'true': np.zeros((5, 5), dtype=int),
            'pred': np.zeros((5, 5), dtype=int)
        }

        self.single_cell = {
            'true': np.array([[0, 1, 1],
                            [0, 1, 1]]),
            'pred': np.array([[0, 1, 1],
                            [0, 1, 1]])
        }

    def test_perfect_match(self):
        """Test metrics calculation with perfectly matching masks."""
        metrics = calculate_object_metrics(
            self.perfect_match['true'],
            self.perfect_match['pred']
        )
        
        self.assertEqual(metrics['mean_iou'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_partial_match(self):
        """Test metrics calculation with partially matching masks."""
        metrics = calculate_object_metrics(
            self.partial_match['true'],
            self.partial_match['pred']
        )
        
        # All cells are matched but with lower IoU
        self.assertLess(metrics['mean_iou'], 1.0)
        self.assertGreater(metrics['mean_iou'], 0.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)

    def test_no_match(self):
        """Test metrics calculation with completely different masks."""
        # Create masks with non-overlapping cells
        metrics = calculate_object_metrics(
            self.no_match['true'],
            self.no_match['pred']
        )
        
        self.assertEqual(metrics['mean_iou'], 0.0)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1_score'], 0.0)

    def test_empty_masks(self):
        """Test metrics calculation with empty masks (no cells)."""
        metrics = calculate_object_metrics(
            self.empty_masks['true'],
            self.empty_masks['pred']
        )
        
        # When both masks are empty, all metrics should be 1.0
        self.assertEqual(metrics['mean_iou'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_single_cell(self):
        """Test metrics calculation with a single cell."""
        metrics = calculate_object_metrics(
            self.single_cell['true'],
            self.single_cell['pred']
        )
        
        self.assertEqual(metrics['mean_iou'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_different_indices(self):
        """Test that cell index values don't affect matching."""
        true_mask = np.array([[1, 1, 0, 2, 2]])
        pred_mask = np.array([[3, 3, 0, 4, 4]])  # Same pattern, different indices
        
        metrics = calculate_object_metrics(true_mask, pred_mask)
        self.assertEqual(metrics['mean_iou'], 1.0)

    def test_iou_threshold(self):
        """Test the effect of IoU threshold on matching."""
        # Create masks with IoU of 0.5
        true_mask = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1]])
        pred_mask = np.array([[0, 1, 1, 0],
                            [0, 1, 1, 0]])
        
        # With threshold = 0.4, the cells should match
        metrics_low = calculate_object_metrics(true_mask, pred_mask, iou_threshold=0.4)
        self.assertEqual(metrics_low['precision'], 1.0)
        self.assertEqual(metrics_low['recall'], 1.0)
        
        # With threshold = 0.6, the cells should not match
        metrics_high = calculate_object_metrics(true_mask, pred_mask, iou_threshold=0.6)
        self.assertEqual(metrics_high['precision'], 0.0)
        self.assertEqual(metrics_high['recall'], 0.0)

class TestBaseSegmenterMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test cases with ImageData objects."""
        # Create a simple 3x3 test case
        self.true_mask = np.array([[[1, 1, 0],
                                   [0, 0, 0],
                                   [2, 2, 0]]])
        
        self.pred_mask = np.array([[[1, 1, 0],
                                   [0, 0, 0],
                                   [2, 2, 0]]])
        
        # Create ImageData objects
        self.image1 = ImageData(
            raw=np.zeros((1, 3, 3)),
            image_id="test1",
            mask=self.true_mask,
            predicted_mask=self.pred_mask
        )
        
        self.image2 = ImageData(
            raw=np.zeros((1, 3, 3)),
            image_id="test2",
            mask=self.true_mask,
            predicted_mask=self.pred_mask
        )

    def test_base_segmenter_metrics(self):
        """Test BaseSegmenter's calculate_metrics method."""
        metrics = BaseSegmenter.calculate_metrics([self.image1, self.image2])
        
        self.assertEqual(metrics['mean_iou'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_missing_mask(self):
        """Test handling of missing masks."""
        # Create image without ground truth mask
        image_no_truth = ImageData(
            raw=np.zeros((1, 3, 3)),
            image_id="test3",
            predicted_mask=self.pred_mask
        )
        
        with self.assertRaises(ValueError):
            BaseSegmenter.calculate_metrics([image_no_truth])

    def test_missing_prediction(self):
        """Test handling of missing predictions."""
        # Create image without predicted mask
        image_no_pred = ImageData(
            raw=np.zeros((1, 3, 3)),
            image_id="test4",
            mask=self.true_mask
        )
        
        with self.assertRaises(ValueError):
            BaseSegmenter.calculate_metrics([image_no_pred])

    def test_dimension_handling(self):
        """Test handling of different mask dimensions."""
        # Create image with 2D masks
        image_2d = ImageData(
            raw=np.zeros((1, 3, 3)),
            image_id="test5",
            mask=self.true_mask[0],  # 2D mask
            predicted_mask=self.pred_mask[0]  # 2D mask
        )
        
        metrics = BaseSegmenter.calculate_metrics([image_2d])
        self.assertEqual(metrics['mean_iou'], 1.0)

if __name__ == '__main__':
    unittest.main()