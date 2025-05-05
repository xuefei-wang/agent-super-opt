from typing import Dict, Tuple, List
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.metrics import compute_surface_dice
from skimage import transform
import torch.nn.functional as F

try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData

from segment_anything import build_sam_vit_b

def medsam_inference(medsam_model, img_embed, box_torch, H, W):
    print("\nInside medsam_inference()")
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
    
    def predict(self, images: ImageData, scaled_boxes, used_for_baseline) -> np.ndarray:
        medsam_model = build_sam_vit_b(device=self.device, checkpoint=self.checkpoint_path)
        medsam_model.to(self.device)
        medsam_model.eval()
       
        resized_imgs = images.raw
        if used_for_baseline:   # also run min-max normalization
            for i, img_1024 in enumerate(resized_imgs):
                resized_imgs[i] = (img_1024 - img_1024.min()) / np.clip(
                    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                )

        img_batch = torch.stack([
            torch.tensor(img).float().permute(2, 0, 1)  # (3, H, W)
            for img in resized_imgs
        ]).to(self.device)  # (B, 3, H, W)
        
        batch_size = 8
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, img_batch.size(0), batch_size):
                print(f"Processing batch {i // batch_size + 1}...")
                batch = img_batch[i:i+batch_size]
                temp_image_embedding = medsam_model.image_encoder(batch)  # (B, 256, 64, 64)
                all_embeddings.append(temp_image_embedding)

        image_embeddings = torch.cat(all_embeddings, dim=0)  # (B, 256, 64, 64)

        scaled_boxes = np.array(scaled_boxes)
        scaled_boxes = torch.tensor(scaled_boxes).float()
        scaled_boxes = scaled_boxes.to(self.device)

        torch.cuda.empty_cache()
        medsam_seg = medsam_inference(medsam_model, image_embeddings, scaled_boxes, 512, 512)
        return medsam_seg
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
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
    
    def preprocess(self, image_data: ImageData) -> ImageData:
        return image_data