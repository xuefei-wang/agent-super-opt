from prompts.task_prompts import TaskPrompts

class CellposeSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a dual-channel nucleus (second channel) and cytoplasm (first channel) segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset of many different cell types. IMPORTANT: The cell images have dimensions (B, L, W, C) = (batch, length, width, channel). To correctly predict masks, the images provided must be in the format of standard ImageData object and must have two channels, the first channel being the cytoplasm and the second channel being the nucleus.

        ```
    """
    
    pipeline_metrics_info = """
        The advantage quantifies how much better this function performs than average (if positive) or how much worse than average (if negative).
        The following metrics are used to evaluate the performance of the pipeline: average_precision, bce_loss
        The average_precision is the average precision score of the pipeline at different IoU thresholds: [0.5, 0.75, 0.9].
        The bce_loss is the binary cross entropy loss of the pipeline for the foreground and background classes (whether a cell is present or not).
        We want to maximize the advantage, increase the average_precision (especially at 0.5), and decrease the bce_loss.
    """

    pipeline_prompt = """
        ```python
            import logging
            from pathlib import Path
            import glob
            import pandas as pd
            import numpy as np
            import torch
            from cellpose.io import imread
            from dotenv import load_dotenv
            from src.data_io import ImageData
            from src.cellpose_segmentation import CellposeTool

            gpu_id = {gpu_id}
            seed = {seed}

            # Set up output directory
            output_dir = Path("{output_dir}")

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

            # Initialize segmenter tool
            segmenter = CellposeTool(model_name="cyto3", device=gpu_id)
            raw_images, gt_masks = segmenter.loadData("{data_path}")

            
            images = ImageData(raw=raw_images, batch_size=16, image_ids=[i for i in range(len(raw_images))])
            
            
            # TODO: fill in your {k_word} preprocessing functions below
            {functions}
            
            preprocessing_fns = [{function_names}]
            
            metrics = []
            
            for preprocessing_fn in preprocessing_fns:
            
                # Apply preprocessing function to images
                images = preprocessing_fn(images)
            
                # Run segmenter
                pred_masks= segmenter.predict(images, batch_size=images.batch_size)
    
                # Calculate metrics
                fn_metrics, fn_losses = segmenter.evaluate(pred_masks, gt_masks)
    
                overall_metrics = {}
                for key, value in fn_metrics.items():
                    if isinstance(value, np.ndarray):
                        overall_metrics[key] = value.tolist()
                    else:
                        overall_metrics[key] = value
                for key, value in fn_losses.items():
                    if isinstance(value, np.ndarray):
                        overall_metrics[key] = value.tolist()
                    else:
                        overall_metrics[key] = value
                metrics.append(overall_metrics)
                logger.info(f"Overall metrics of function {preprocessing_fn.__name__}: ", overall_metrics)
            
            compute_reward = lambda m: -m["bce_loss"]
            
            group_baseline = np.mean(list(map(compute_reward, metrics)))
            
            metrics = [{{**m, "advantage": compute_reward(m) - group_baseline}} for m in metrics]
           
            ```
    """

    task_details = """
    All of you should work together to write {k_word} preprocessing functions to improve segmentation performance using OpenCV functions.
    1. Based on previous preprocessing functions and their respective advantages (provided below), suggest {k_word} new unique preprocessing functions that maximize the advantages using OpenCV functions (APIs provided below). Remember, the bigger the advantage for a particular function, the better it performed than average.
    2. Plug each of the {k_word} preprocessing functions into the pipeline and run the segmenter to calculate the performance metrics and advantages of each one, using the provided code snippet.
    3. Save the {k_word} newly proposed preprocessing functions, their performance metrics, and their advantages in the function bank, using the provided script.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    6. Do not terminate the conversation until the new preprocessing functions are evaluated.
    7. Extremely important: Do not terminate the conversation until the new preprocessing functions are evaluated AND they must be written to the function bank by calling the write_results function.
    8. Recall, this is not a stateful kernel, so all functions, imports, etc. must be provided in the script to be executed.
    """


    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, grpo_k, grpo_k_word):
        super().__init__(
            gpu_id=gpu_id,
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
        # Create a copy of the pipeline_prompt string
        prompt = super().run_pipeline_prompt()
        
        # Replace placeholders manually
        prompt = prompt.replace("{function_bank_path}", str(self.function_bank_path))
        prompt = prompt.replace("{output_dir}", '/'.join(self.function_bank_path.split('/')[:-1]))
        
        return prompt
    
    def save_function_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path, output_dir = '/'.join(self.function_bank_path.split('/')[:-1]))

    def task_details_prompt(self) -> str:
        return super().task_details_prompt()