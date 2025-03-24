from prompts.task_prompts import TaskPrompts

class CellSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a single-channel nuclear segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset covering
        five different cell lines (NIH-3T3, HeLa-S3, HEK293, RAW 264.7, and PC-3).
        ```
    """
    
    summary_prompt = """
        Summarize the results as a python dictionary, including the newly proposed preprocessing function and its average performance metrics.
        Follow the format:
        {
        "mean_iou": ...,
        "precision": ...,
        "recall": ...,
        "f1_score": ...,
        "preprocessing_function": "
            ```python
            YOUR_CODE_HERE
            ```
            ",
        }
        """
    
    save_to_function_bank_prompt = """
        ```python
        import inspect
        import json

        def write_results(preprocessing_fn, metrics_dict):
            '''
            Write the results of evaluation to the function bank JSON.
            
            Requires:
            preprocessing_fn: the function
            metrics_dict: the metrics dictionary
            '''
            
            with open('output/preprocessing_func_bank.json', 'r') as file:
                json_array = json.load(file)

            with open('output/preprocessing_func_bank.json', 'w') as file:
                json_data = metrics_dict
                json_data["preprocessing_function"] = inspect.getsource(preprocessing_fn)
                json_array.append(json_data)
                json.dump(json_array, file)
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

            # TODO: add your preprocessing function here
            images = preprocess_images(images)

            # Initialize segmenter
            segmenter = MesmerSegmenter()

            # Run segmenter
            results = segmenter.predict(images)

            # Calculate metrics
            metrics = calculate_metrics(results.masks, results.predicted_masks)
            df = pd.DataFrame(metrics)
            overall_metrics = df.mean().to_dict()
            logger.info("Overall metrics: ", overall_metrics)

            ```
    """

    task_details = """
    All of you should work together to write a preprocessing function to improve segmentation performance using OpenCV functions.
    1. Based on previous preprocessing functions and their performance (provided below), suggest a new preprocessing function using OpenCV functions (APIs provided below).
    2. Plug the preprocessing function into the pipeline and run the segmenter to calculate the performance metrics, using the provided code snippet.
    3. Save the newly proposed preprocessing function and its performance metrics in the function bank, using the provided script.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    6. Do not terminate the conversation until the new preprocessing function is evaluated.
    """


    def __init__(self, gpu_id, seed, dataset_path, function_bank_path):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            summary_prompt=self.summary_prompt,
            task_details=self.task_details,
            function_bank_path=function_bank_path,
        )
    
    def run_pipeline_prompt(self) -> str:
        return self.pipeline_prompt.format(gpu_id=self.gpu_id, seed=self.seed, data_path=self.dataset_path)
    
    def save_function_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path)