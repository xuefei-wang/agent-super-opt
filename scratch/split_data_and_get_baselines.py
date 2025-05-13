import os
import sys
import importlib
import time

import numpy as np
import pickle
from skimage import transform

project_root = os.path.abspath(os.path.join(os.getcwd(), "..")) # scratch folder
if project_root not in sys.path:
    sys.path.append(project_root)

import importlib
from src.medsam_segmentation import MedSAMTool
from src.data_io import ImageData
importlib.reload(importlib.import_module('src.medsam_segmentation'))

# ====
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
# ====

# randomly sample 5 xray images
for modality in ["xray", "dermoscopy"]:
    query_name = "X-Ray" if modality == "xray" else "Dermoscopy"
    
    imgs_2d_and_3d = os.listdir("/home/sstiles/sci-agent/data/imgs")  # 3076
    imgs_2d = [f for f in imgs_2d_and_3d if f.startswith('2D')] # 2803
    imgs_2d_modality = [f for f in imgs_2d if query_name in f]    # 50
    print(f"{len(imgs_2d_modality)} images in {modality} modality")

    np.random.seed(42)
    half_len = len(imgs_2d_modality) // 2
    val_filenames_bank  = np.random.choice(imgs_2d_modality, size=half_len, replace=False)
    test_filenames_bank = [f for f in imgs_2d_modality if f not in val_filenames_bank]
                    
    for exp_type in ["test", "val"]:
        file_str = f"{modality}_{exp_type}"
        print("Starting experiment", file_str)
        if exp_type == "val":
            filebank = val_filenames_bank
        else:
            filebank = test_filenames_bank
        
        # ================ Unpack images ================
        unpack_start_time = time.time()
        print(f"\nUnpacking {len(filebank)} images...")
        raw_imgs, raw_boxes, raw_gts = [], [], []
        for i, img_filename in enumerate(filebank):
            img_data = np.load(f"/home/sstiles/sci-agent/data/imgs/{img_filename}")
            mask_data = np.load(f"/home/sstiles/sci-agent/data/gts/{img_filename}")
            
            image, boxes, nonbinary_mask = img_data['imgs'], img_data["boxes"], mask_data['gts']
            num_masks = 0
            for box, mask in zip(boxes, _get_binary_masks(nonbinary_mask)):
                x1, y1, x2, y2 = box
                box_string = f"[{x1},{y1},{x2},{y2}]"
                raw_imgs.append(image)
                raw_boxes.append(box_string)
                raw_gts.append(mask)
                num_masks += 1
            
            print(f"Processed idx {i}: {img_filename} -> {num_masks} masks")
        print(f"Unpacked {len(raw_imgs)} images in {time.time() - unpack_start_time:.2f} seconds")

        # ================ Resize images ================
        print("Resizing images...")
        resize_start_time = time.time()

        if exp_type == "val":
            random_25_indices = np.random.choice(len(raw_imgs), size=25, replace=False)
            imgs_to_resize = [raw_imgs[i] for i in random_25_indices]
            boxes_to_resize = [raw_boxes[i] for i in random_25_indices]
            gts_to_use = [raw_gts[i] for i in random_25_indices]
        else:   # if it's test, we use all unpacked triplets from the file bank
            imgs_to_resize = raw_imgs
            boxes_to_resize = raw_boxes
            gts_to_use = raw_gts
        
        resized_imgs, resized_boxes = [], []
        resized_gts = gts_to_use
        for i, (img_np, box_str) in enumerate(zip(imgs_to_resize, boxes_to_resize)):
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np

            H, W, _ = img_3c.shape

            # Resize image to 1024x1024
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)

            img_1024 = img_1024 / 255.0
            resized_imgs.append(img_1024)

            # Scale box to 1024x1024
            box_np = np.array([[int(x) for x in box_str[1:-1].split(',')]])
            box_scaled = box_np / np.array([W, H, W, H]) * 1024
            resized_boxes.append(box_scaled)

            print(f"file {i} | og img shape {img_np.shape} | box_str {box_str} -> {box_scaled.shape}")
        print(f"Finished resizing images in {time.time() - resize_start_time:.2f} seconds\n")

        resized_filepath = f"/home/sstiles/sci-agent/scratch/resized_{file_str}.pkl"
        # if file doesn't exist, create it
        if not os.path.exists(resized_filepath):
            os.makedirs(os.path.dirname(resized_filepath), exist_ok=True)
        with open(resized_filepath, "wb") as f:
            pickle.dump((resized_imgs, resized_boxes, resized_gts), f)
        print(f"Saved resized data to {resized_filepath}\n")

        # ================ Get baseline ================
        baseline_start_time = time.time()
        log_file = f"/home/sstiles/sci-agent/scratch/baseline_{file_str}.txt"
        # if file doesn't exist, create it
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        print("Writing baseline to", log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(log_file, "w") as log_file_obj:
            sys.stdout = log_file_obj
            sys.stderr = log_file_obj

            segmenter = MedSAMTool(gpu_id=2, checkpoint_path="/home/sstiles/sci-agent/data/medsam_vit_b.pth")

            used_imgs = resized_imgs
            used_boxes = resized_boxes
            used_masks = resized_gts

            images = ImageData(
                raw=used_imgs,
                batch_size=min(8, len(used_imgs)),
                image_ids=[i for i in range(len(used_imgs))],
                masks=used_masks,
                predicted_masks=used_masks,
            )

            pred_masks = segmenter.predict(images, used_boxes, used_for_baseline=True)
            segmenter.evaluate(pred_masks, used_masks)

        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Finished running baseline in {time.time() - baseline_start_time:.2f} seconds")
