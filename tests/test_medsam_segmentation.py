import unittest
from src.medsam_segmentation import MedSAMTool, prepare_image_data
import argparse
import sys

'''
To run from the project root directory:

CPU: python -m tests.test_medsam_segmentation --checkpoint_path {checkpoint_path} --data_path {data_path}
GPU: python -m tests.test_medsam_segmentation --device {gpu_id}
'''

class TestMedSAMSegmentation(unittest.TestCase):
    def test_pipeline(self):
        """
        Tests that the prediction and evaluation pipeline completes without errors.
        """    
        medsam_tool = MedSAMTool(gpu_id=self.device, checkpoint_path=self.checkpoint_path)
        images = prepare_image_data(self.data_path, num_files=1, batch_size=1)
        pred_masks = medsam_tool.predict(images)
        gt_masks_list = images.masks
        losses = medsam_tool.evaluate(pred_masks, gt_masks_list)
        print('losses', losses)

        self.assertIn('dice_loss', losses)

def parse_args():
    parser = argparse.ArgumentParser(description='Run MedSAM segmentation tests with custom parameters')
    parser.add_argument('--device', type=int, default=0,
                      help='GPU device ID to run on.')
    parser.add_argument('--checkpoint_path', type=str, default='/workspace/data/medsam_vit_b.pth', help='Path to the model checkpoint file.')
    parser.add_argument('--data_path', type=str, default='/workspace/data', help='Path to dataset.')
    
    # This allows unittest arguments to be passed through
    args, unittest_args = parser.parse_known_args()
    sys.argv[1:] = unittest_args
    
    return args

if __name__ == '__main__':
    args = parse_args()
    TestMedSAMSegmentation.device = args.device
    TestMedSAMSegmentation.checkpoint_path = args.checkpoint_path
    TestMedSAMSegmentation.data_path = args.data_path
    unittest.main()
