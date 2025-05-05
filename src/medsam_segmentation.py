import os
from typing import Dict, Tuple, List
import numpy as np
import torch
import glob
from monai.metrics import DiceMetric
from monai.metrics import compute_surface_dice
from skimage import transform
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData

from segment_anything import build_sam_vit_b
from cv2 import imread

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


def preprocess_stage2(resized_imgs, medsam_model, device):
    print("\nInside preprocess_stage2()")
    # Convert to tensor batch
    img_batch = torch.stack([
        torch.tensor(img).float().permute(2, 0, 1)  # (3, H, W)
        for img in resized_imgs
    ]).to(device)  # (B, 3, H, W)
    
    batch_size = 8
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, img_batch.size(0), batch_size):
            print(f"Processing batch {i // batch_size + 1}...")
            batch = img_batch[i:i+batch_size]
            image_embeddings = medsam_model.image_encoder(batch)  # (b, 256, 64, 64)
            all_embeddings.append(image_embeddings)

    # Concatenate all the batch outputs into a single tensor
    return torch.cat(all_embeddings, dim=0)  # (B, 256, 64, 64)

def medsam_batch(medsam_model, img_embed, box_torch, H, W):
    print("\nInside medsam_batch()")
    batch_size = 8
    all_predictions = []

    for i in range(0, img_embed.size(0), batch_size):
        print(f"Processing batch {i // batch_size + 1}...")

        img_embed_batch = img_embed[i:i + batch_size]
        box_torch_batch = box_torch[i:i + batch_size]

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch_batch,
            masks=None,
        )

        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed_batch,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (B, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)
        
        for j in range(low_res_pred.shape[0]):
            pred = low_res_pred[j].squeeze().detach().cpu().numpy()  # (H, W)
            medsam_seg = (pred > 0.5).astype(np.uint8)
            all_predictions.append(torch.from_numpy(medsam_seg).unsqueeze(0))

    return torch.cat(all_predictions, dim=0)  # (N, H, W)

class MedSAMTool():
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

    def save_resized_imgs(self, images: ImageData, save_path) -> List[np.ndarray]:
        print("Inside save_resized_images...")
        medsam_model = build_sam_vit_b(device=self.device, checkpoint=self.checkpoint_path)
        medsam_model.to(self.device)
        medsam_model.eval()

        raw_imgs = images.raw
        raw_boxes = [self._get_bounding_box(mask_np) for mask_np in images.predicted_masks]
        resized_imgs, scaled_boxes = preprocess_stage1(images.raw, raw_boxes)
        with open(save_path, "wb") as f:
            pickle.dump((resized_imgs, scaled_boxes), f)
    
    def saved_new_predict(self, images: ImageData, scaled_boxes, used_for_baseline) -> np.ndarray:
        medsam_model = build_sam_vit_b(device=self.device, checkpoint=self.checkpoint_path)
        medsam_model.to(self.device)
        medsam_model.eval()
       
        resized_imgs = images.raw
        if used_for_baseline:   # also run min-max normalization
            for i, img_1024 in enumerate(resized_imgs):
                resized_imgs[i] = (img_1024 - img_1024.min()) / np.clip(
                    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                )

        image_embeddings = preprocess_stage2(resized_imgs, medsam_model, device=self.device)

        scaled_boxes = np.array(scaled_boxes)
        scaled_boxes = torch.tensor(scaled_boxes).float()
        scaled_boxes = scaled_boxes.to(self.device)

        torch.cuda.empty_cache()
        medsam_seg = medsam_batch(medsam_model, image_embeddings, scaled_boxes, 512, 512)
        return medsam_seg
    
    def new_evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        resized_preds = []
        for i in range(len(pred_masks)):
            H, W = gt_masks[i].shape
            resized_pred = transform.resize(
                pred_masks[i].cpu().numpy(), (H, W), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            resized_preds.append(resized_pred)
            
        print("\nEvaluating predictions...")
        spacing= (1.0, 1.0)  # Assuming isotropic spacing for simplicity
        tolerance = 2.0
        
        total_dsc, total_nsd = 0, 0
        for i, (pred, gt) in enumerate(zip(resized_preds, gt_masks)):
            gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

            dice_metric = DiceMetric(include_background=False, reduction="mean")
            immediate_dsc_metric = dice_metric(pred_tensor, gt_tensor)

            immediate_nsd_metric = compute_surface_dice(pred_tensor, gt_tensor, class_thresholds=[tolerance], spacing=spacing)
            print(f"{i} | DSC: {round(immediate_dsc_metric.item(), 6)} | NSD: {round(immediate_nsd_metric.item(), 6)}")
            
            total_dsc += immediate_dsc_metric
            total_nsd += immediate_nsd_metric

        print("\n=======================")
        print(f"Average DSC metric: {round(total_dsc.item() / len(pred_masks), 6)}")
        print(f"Average NSD metric: {round(total_nsd.item() / len(pred_masks), 6)}")
        return {"dsc_metric": total_dsc.item() / len(pred_masks),
                "nsd_metric": total_nsd.item() / len(pred_masks)}
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate detections against ground truth using the model output.

        Args:
            pred_masks: ndarray, predicted masks
            gt_masks: ndarray, binary ground truth masks
        
        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
                - dsc_metric: the dice similarity coefficient (DSC) score
                - nsd_metric: the normalized surface distance (NSD) score
        """
        print("\n=======================")
        print("Evaluating predictions...")
        spacing= (1.0, 1.0)  # Assuming isotropic spacing for simplicity
        tolerance = 2.0
        
        total_dsc, total_nsd = 0, 0
        for i, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

            dice_metric = DiceMetric(include_background=False, reduction="mean")
            immediate_dsc_metric = dice_metric(pred_tensor, gt_tensor)

            immediate_nsd_metric = compute_surface_dice(pred_tensor, gt_tensor, class_thresholds=[tolerance], spacing=spacing)
            print(f"{i} | DSC: {round(immediate_dsc_metric.item(), 6)} | NSD: {round(immediate_nsd_metric.item(), 6)}")
            
            total_dsc += immediate_dsc_metric
            total_nsd += immediate_nsd_metric

        print("\n=======================")
        print(f"Average DSC metric: {round(total_dsc.item() / len(pred_masks), 6)}")
        print(f"Average NSD metric: {round(total_nsd.item() / len(pred_masks), 6)}")
        return {"dsc_metric": total_dsc.item() / len(pred_masks),
                "nsd_metric": total_nsd.item() / len(pred_masks)}

    def preprocess(self, image_data: ImageData) -> ImageData:
        return image_data

    def visualize(self, image_data, pred_masks, gt_masks):
        _, axes = plt.subplots(3, 2, figsize=(24, 24))
        for i, (image, pred_mask, gt_mask) in enumerate(zip(image_data.raw, pred_masks, gt_masks)):
            # Plot predicted mask
            ax = axes[i, 0]
            ax.imshow(image)
            ax.imshow(pred_mask, alpha=0.5, cmap='gray')
           
           # Plot bounding box
            box_string = self._get_bounding_box(gt_mask)
            x1, y1, x2, y2 = map(int, box_string.strip('[]').split(','))
            width, height = x2 - x1, y2 - y1
            rect_pred = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_pred)
            ax.set_title(f"Image {i+1}: Predicted Mask")
            ax.axis('off')

            # Plot ground truth mask
            ax = axes[i, 1]
            ax.imshow(image)
            ax.imshow(gt_mask, alpha=0.5, cmap='gray')
            rect_gt = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_gt)
            ax.set_title(f"Image {i+1}: Ground Truth Mask")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def _get_binary_masks(nonbinary_mask):
        """ 
        Given nonbinary mask which encodes N masks, return N binary masks which
        should encode the same information.
        
        Parameters:
            - nonbinary_mask: ndarray of shape (H, W)
        Returns:
            - binary_masks: ndarray of shape (N, H, W)
        """
        binary_masks = []
        for i in np.unique(nonbinary_mask)[1:]:
            binary_mask = (nonbinary_mask == i).astype(np.uint8)
            binary_masks.append(binary_mask.copy())
        binary_masks = np.stack(binary_masks, axis=0)
        return binary_masks

def prepare_image_data(data_path, num_files, batch_size=8) -> ImageData:
    """
    Construct an ImageData object from the MedSAM dataset with NPZ format.

    Args:
        data_path: str, path to the dataset which includes image and mask
            directories.
        num_files: int, number of files to load
        batch_size: int, batch size for the ImageData object
    
    Returns:
        ImageData object containing the images and masks
    """
    img_path = os.path.join(data_path, 'imgs')
    mask_path = os.path.join(data_path, 'gts')

    img_files = sorted(glob.glob(os.path.join(img_path, '*')))[:num_files]
    mask_files = sorted(glob.glob(os.path.join(mask_path, '*')))[:num_files]

    raw_images, raw_boxes, raw_masks = [], [], []
    for img_npz_file, mask_npz_file in zip(img_files, mask_files):
        img_data, mask_data = np.load(img_npz_file), np.load(mask_npz_file)  
        
        image, boxes, nonbinary_mask = img_data['imgs'], img_data["boxes"], mask_data['gts']
        binary_masks = _get_binary_masks(nonbinary_mask)
        
        for box, mask in zip(boxes, binary_masks):
            x1, y1, x2, y2 = box
            box_string = f"[{x1},{y1},{x2},{y2}]"
            
            raw_images.append(image)
            raw_boxes.append(box_string)
            raw_masks.append(mask)

    images = ImageData(raw=raw_images,
                    batch_size=batch_size,
                    image_ids=[i for i in range(len(raw_images))],
                    masks=raw_masks,
                    predicted_masks=raw_masks)
    return images


def prepare_image_data_png(data_path, num_files, batch_size=8) -> ImageData:
    """
    Construct an ImageData object from the MedSAM dataset with PNG format.

    Args:
        data_path: str, path to the dataset which includes image and mask
            directories.
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