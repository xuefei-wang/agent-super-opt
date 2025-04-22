from prompts.task_prompts import TaskPrompts

class MedSAMSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a chest X-ray image lung segmentation dataset. Some images
        feature lungs with tuberculosis, while others are normal. The images have
        dimensions (H, W, C) = (height, width, channel).
        ```
    """

    pipeline_metrics_info = """
        The advantage quantifies how much better this function performs than average (if positive) or how much worse than average (if negative). We want to maximize the advantage.
        The following metrics are used to evaluate the performance of the pipeline: dice_loss.
        The dice loss is the dice similarity coefficient (DSC) score of the pipeline.
    """

    pipeline_prompt = """
        ```python
            import logging
            from pathlib import Path
            import glob
            import pandas as pd
            import numpy as np
            from dotenv import load_dotenv
            from src.data_io import ImageData
            from src.medsam_segmentation import MedSAMTool
            from cv2 import imread
            import os

            gpu_id = {gpu_id}
            checkpoint_path = "{checkpoint_path}"
            seed = {seed}

            # Set up output directory
            output_dir = Path("output")

            # Set up logging
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler("app.log"),
                    logging.StreamHandler()  # This keeps console logging
                ]
            )
            logger = logging.getLogger(__name__)


            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load data
            def _get_binary_masks(nonbinary_mask):
                '''
                Given nonbinary mask which encodes N masks, return N binary masks which
                should encode the same information.
                
                Parameters:
                    - nonbinary_mask: ndarray of shape (H, W)
                Returns:
                    - binary_masks: ndarray of shape (N, H, W)
                '''
                binary_masks = []
                for i in np.unique(nonbinary_mask)[1:]:
                    binary_mask = (nonbinary_mask == i).astype(np.uint8)
                    binary_masks.append(binary_mask.copy())
                binary_masks = np.stack(binary_masks, axis=0)
                return binary_masks
            data_path = '{data_path}'
            img_path = os.path.join(data_path, 'imgs')
            mask_path = os.path.join(data_path, 'gts')

            num_files = 1
            img_files = sorted(glob.glob(os.path.join(img_path, '*')))[:num_files]
            mask_files = sorted(glob.glob(os.path.join(mask_path, '*')))[:num_files]

            raw_images, raw_boxes, raw_masks = [], [], []
            for img_npz_file, mask_npz_file in zip(img_files, mask_files):
                img_data, mask_data = np.load(img_npz_file), np.load(mask_npz_file)  
                
                image, boxes, nonbinary_mask = img_data['imgs'], img_data["boxes"], mask_data['gts']
                binary_masks = _get_binary_masks(nonbinary_mask)
                
                for box, mask in zip(boxes, binary_masks):
                    x1, y1, x2, y2 = box
                    box_string = "[" + ",".join(map(str, [x1, y1, x2, y2])) + "]"
                    
                    raw_images.append(image)
                    raw_boxes.append(box_string)
                    raw_masks.append(mask)

            images = ImageData(raw=raw_images,
                            batch_size=batch_size,
                            image_ids=[i for i in range(len(raw_images))],
                            masks=raw_masks,
                            predicted_masks=raw_masks)
            
            # Initialize segmenter
            segmenter = MedSAMTool(gpu_id={gpu_id}, checkpoint_path={checkpoint_path})
            
            # TODO: fill in your {k_word} preprocessing functions below
            {functions}
            
            preprocessing_fns = [{function_names}]
            
            metrics = []
            
            for preprocessing_fn in preprocessing_fns:
            
                # Apply preprocessing function to images
                images = preprocessing_fn(images)

                # Run segmenter
                pred_masks = segmenter.predict(images)
    
                # Calculate Metrics
                losses = segmenter.evaluate(pred_masks, images.masks)
    
                df = pd.DataFrame([losses])
                overall_losses = df.mean().to_dict()
                metrics.append(overall_losses)
                logger.info(f"Overall losses of function {preprocessing_fn.__name__}: ", overall_losses)
                
            compute_reward = lambda m: -m["dice_loss"]
            
            group_baseline = np.mean(list(map(compute_reward, metrics)))
            
            metrics = [{{**m, "advantage": compute_reward(m) - group_baseline}} for m in metrics]
            ```
    """

    task_details = """
    All of you should work together to write {k_word} preprocessing functions to improve segmentation performance using OpenCV functions.
    1. Based on previous preprocessing functions and their respective advantages (provided below), suggest {k_word} new unique preprocessing functions that maximize the advantages using OpenCV functions (APIs provided below). Remember, the bigger the advantage for a particular function, the better it performed than average. Also, the images after preprocessing must still conform to the format specified in the ImageData API.
    2. Plug each of the {k_word} preprocessing functions into the pipeline and run the segmenter to calculate the performance metrics and advantages of each one, using the provided code snippet. Make sure the loading of images, preprocessing function implementations, and all other pipeline workflow code are contained in the SAME code block.
    3. Save the {k_word} newly proposed preprocessing functions, their performance metrics, and their advantages in the function bank, using the provided script.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    6. Do not terminate the conversation until the new preprocessing functions are evaluated and the advantages and numerical loss metric(s) are logged.
    """


    def __init__(self, gpu_id, checkpoint_path, seed, dataset_path, function_bank_path, grpo_k, grpo_k_word):
        super().__init__(
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            summary_prompt=None,
            pipeline_prompt=self.pipeline_prompt,
            task_details=self.task_details,
            function_bank_path=function_bank_path,
            pipeline_metrics_info=self.pipeline_metrics_info,
            grpo_k=grpo_k,
            grpo_k_word=grpo_k_word,
        )
    
    def run_pipeline_prompt(self) -> str:
        prompt = super().run_pipeline_prompt()
        return prompt.format(checkpoint_path=self.checkpoint_path)
    
    def save_function_prompt(self) -> str:
        return super().save_function_prompt()

    def task_details_prompt(self) -> str:
        return super().task_details_prompt()