import os
from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob
import monai

try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData

try:
    from tools import BaseSegmenter
except ImportError:
    from src.tools import BaseSegmenter

try:
    from utils import set_gpu_device
except ImportError:
    from src.utils import set_gpu_device

from medsam import medsam_inference, show_box, show_mask, preprocess, visualize_results
from segment_anything import build_sam_vit_b
from cv2 import imread

# MedSAM_CKPT_PATH = '/workspace/data/medsam_vit_b.pth'
# MedSAM_CKPT_PATH = 'work_dir/MedSAM/medsam_vit_b.pth'

class MedSAMTool(BaseSegmenter):
    """
    MedSAMTool is a class that provides a simple interface for the MedSAM model
    """

    def __init__(self, checkpoint_path, model_name='medsam', gpu_id: int = 0, **kwargs):
        self.checkpoint_path = checkpoint_path
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            print(f"Warning: GPU ID {gpu_id} is not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            
        self.kwargs = kwargs
        
    def _get_bounding_box(self, mask):
        """
        Calculate the bounding box from the ground truth mask.
        
        Args:
            mask (numpy.ndarray): Ground truth mask with shape (H, W).
            
        Returns:
            str: Bounding box in the format "[x_min, y_min, x_max, y_max]".
        """
        rows, cols = np.where(mask > 0)
        
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        return f"[{x_min},{y_min},{x_max},{y_max}]"

    def predict(self, images: ImageData) -> np.ndarray:
        """
        Predict masks for a batch of images. According to the MedSAM paper,
        the MedSAM model requires that input is resized to a uniform size
        of 1024 x 1024 x 3.

        Args:
            images: ImageData object containing a batch of images. Contains 
                'raw' and 'masks' attributes in the format of standard 
                ImageData object [B, H, W, C].
            
        Returns:
            List[np.ndarray]: A list of binary masks corresponding to each of 
                the input images.
        """
        medsam_model = build_sam_vit_b(device=self.device, checkpoint=self.checkpoint_path)
        medsam_model.to(self.device)
        medsam_model.eval()

        all_masks = []
        for img_np, mask_np in zip(images.raw, images.masks):
            # preprocess
            box = self._get_bounding_box(mask_np)
            image_embedding, box_1024, H, W, _, _ = preprocess(medsam_model, img_np, box, device=self.device)
            mask = medsam_inference(medsam_model, image_embedding, box_1024, H, W)   
            all_masks.append(mask)   
        return all_masks
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate detections against ground truth using the model output.

        Args:
            pred_masks: ndarray, predicted masks
            gt_masks: ndarray, binary ground truth masks
        
        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
                - dice_loss: the dice similarity coefficient (DSC) score
        """
        total_dice_loss = 0
        for pred, gt in zip(pred_masks, gt_masks):
            pred_tensor = torch.tensor(pred, dtype=torch.float32)
            gt_tensor = torch.tensor(gt / 255.0, dtype=torch.float32)

            dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
            total_dice_loss += dice_loss(pred_tensor, gt_tensor)

        return {"dice_loss": total_dice_loss.item() / len(pred_masks)}

    def preprocess(self, image_data: ImageData) -> ImageData:
        return image_data

def prepare_image_data(data_path, num_files, batch_size=8) -> ImageData:
    """
    Construct an ImageData object from the MedSAM dataset.

    Args:
        data_path: str, path to the dataset which includes image and mask
            directories
        num_files: int, number of files to load
        batch_size: int, batch size for the ImageData object
    
    Returns:
        ImageData object containing the images and masks
    """
    img_path = os.path.join(data_path, 'CXR_png')
    mask_path = os.path.join(data_path, 'masks')

    img_files = sorted(glob.glob(os.path.join(img_path, '*')))[:num_files]
    mask_files = sorted(glob.glob(os.path.join(mask_path, '*')))[:num_files]

    raw_images = [imread(f) for f in img_files]
    raw_masks = [imread(f) for f in mask_files]
    raw_masks = [mask[:, :, 0] if len(mask.shape) == 3 else mask for mask in raw_masks]

    images = ImageData(raw=raw_images,
                    batch_size=batch_size,
                    image_ids=[i for i in range(num_files)],
                    masks=raw_masks,
                    predicted_masks=raw_masks)
    return images

if __name__ == "__main__":
    medsam_tool = MedSAMTool()
    images = prepare_image_data(data_path="data/medsam_data", num_files=2, batch_size=1)
    pred_masks = medsam_tool.predict(images)
    losses = medsam_tool.evaluate(pred_masks, images.masks)
    print(losses)
    print('done')