from prompts.task_prompts import TaskPrompts

class CellSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a single-channel nuclear segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset covering
        five different cell lines (NIH-3T3, HeLa-S3, HEK293, RAW 264.7, and PC-3).
        ```
    """

    pipeline_prompt = """
        ```python
            import numpy as np
            import logging
            import pandas as pd
            from pathlib import Path

            import torch
            import tensorflow as tf

            from src.utils import set_gpu_device
            from src.data_io import NpzDataset
            from src.segmentation import MesmerSegmenter, calculate_metrics

            gpu_id = {gpu_id}
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

            # Set GPU device
            set_gpu_device(gpu_id)

            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            tf.random.set_seed(seed)

            # Load data
            data_path = "{data_path}"
            dataset = NpzDataset(data_path)
            indices = np.random.choice(len(dataset), size=5, replace=False)
            images = dataset.load(indices)
            
            # Initialize segmenter
            segmenter = MesmerSegmenter()

            # TODO: fill in your {k_word} preprocessing functions below
            {functions}
            
            preprocessing_fns = [{function_names}]
            
            metrics = []
            
            for preprocessing_fn in preprocessing_fns:
            
                # Apply preprocessing function to images
                images = preprocessing_fn(images)
            
                # Run segmenter
                results = segmenter.predict(images)
    
                # Calculate metrics
                batch_metrics = calculate_metrics(results.masks, results.predicted_masks)
                df = pd.DataFrame(batch_metrics)
                overall_metrics = df.mean().to_dict()
                metrics.append(overall_metrics)
                logger.info(f"Overall metrics of function {preprocessing_fn.__name__}: ", overall_metrics)
            
            compute_reward = lambda m: m["f1_score"]
            
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
            grpo_k=grpo_k,
            grpo_k_word=grpo_k_word,
        )
    
    def run_pipeline_prompt(self) -> str:
        return super().run_pipeline_prompt()
    
    def save_function_prompt(self) -> str:
        return super().save_function_prompt()

    def task_details_prompt(self) -> str:
        return super().task_details_prompt()