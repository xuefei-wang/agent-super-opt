import unittest
import numpy as np
import argparse
import sys
from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
from cellpose.io import imread
from dotenv import load_dotenv
import glob
from numpy.testing import assert_array_equal
### To run from project root directory: python -m tests.test_cellpose_segmentation

class TestCellposeSegmentation(unittest.TestCase):
    # Class variables to store command line arguments
    data_path = '/home/alexfarhang/data/cellpose/train/'
    device = 1
    model_name = "cyto3"
    num_files = 8

    def test_pipeline(self):
        '''Test if prediction and evaluation pipeline completes without errors'''
        load_dotenv()

        cellpose_tool = CellposeTool(model_name=self.model_name, device=self.device)
        f_path = self.data_path
        files = sorted(glob.glob(f_path + '*_img.png'))[:self.num_files]
        raw_images = [imread(f) for f in files]

        gt_masks = [imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files]
        gt_masks = [np.expand_dims(mask, axis=2) for mask in gt_masks]

        images = ImageData(raw=raw_images,
                        batch_size=8,
                        image_ids=[i for i in range(self.num_files)],
                        masks=gt_masks)
        
        def preprocess_images(images: ImageData) -> ImageData:
            return images

        images = preprocess_images(images)        

        pred_masks, flows, styles, imgs = cellpose_tool.predict(images, batch_size=images.batch_size)
        # gt_masks_list = [images.masks[i] for i in range(images.masks.shape[0])] 
        metrics, losses = cellpose_tool.evaluate(pred_masks, images.masks)
        print(metrics)
        print(losses)
            
        self.assertIn('average_precision', metrics)
        self.assertIn('bce_loss', losses)
        # assert_array_equal(metrics['average_precision'], np.array([0.9464578, 0.8065034, 0.3700052], dtype=np.float32))
        # self.assertEqual(losses['bce_loss'], 0.6078091263771057)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Cellpose segmentation tests with custom parameters')
    parser.add_argument('--data_path', type=str, default='/home/alexfarhang/data/cellpose/train/',
                      help='Path to the dataset directory')
    parser.add_argument('--device', type=int, default=1,
                      help='GPU ID to use.')
    parser.add_argument('--num_files', type=int, default=8,
                      help='Number of files to test')
    
    # This allows unittest arguments to be passed through
    args, unittest_args = parser.parse_known_args()
    sys.argv[1:] = unittest_args
    
    return args


if __name__ == '__main__':
    # Parse arguments and set class variables
    args = parse_args()
    TestCellposeSegmentation.data_path = args.data_path
    TestCellposeSegmentation.device = args.device
    TestCellposeSegmentation.num_files = args.num_files
    
    unittest.main()