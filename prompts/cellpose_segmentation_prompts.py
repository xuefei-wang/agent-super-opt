from prompts.task_prompts import TaskPrompts
_PREPROCESSING_FUNCTION_PLACEHOLDER = "# --- CODEGEN_PREPROCESSING_FUNCTIONS_INSERT ---"
import textwrap
import os

class CellposeSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a dual-channel nucleus (second channel) and cytoplasm (first channel) segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset of many different cell types. IMPORTANT: The cell images have dimensions (B, L, W, C) = (batch, length, width, channel). To correctly predict masks, the images provided must be in the format of standard ImageData object and must have two channels, the first channel being the cytoplasm and the second channel being the nucleus.

        ```
    """
    
    summary_prompt = """
        Summarize the results as a python dictionary, including the newly proposed preprocessing function and its average performance metrics.
        Follow the format:
        {
        "average_precision": ...,
        "bce_loss": ...,
        "preprocessing_function": "
            ```python
            YOUR_CODE_HERE
            ```
            ",
        }
        """
    
    pipeline_metrics_info = """
        The following metrics are used to evaluate the performance of the pipeline: average_precision, bce_loss
        The average_precision is the average precision score of the pipeline at different IoU thresholds: [0.5, 0.75, 0.9].
        The bce_loss is the binary cross entropy loss of the pipeline for the foreground and background classes (whether a cell is present or not).
        We want to increase the average_precision (especially at 0.5) and decrease the bce_loss.
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
            
            with open('{function_bank_path}', 'r') as file:
                json_array = json.load(file)

            with open('{function_bank_path}', 'w') as file:
                json_data = metrics_dict
                json_data["preprocessing_function"] = inspect.getsource(preprocessing_fn)
                json_array.append(json_data)
                json.dump(json_array, file)
        ```
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

            # TODO: add your preprocessing function here
            # def preprocess_images(images: ImageData) -> ImageData:
            #   YOUR CODE HERE

            images = preprocess_images(images)

            # Run segmenter
            pred_masks= segmenter.predict(images, batch_size=images.batch_size)

            # Calculate metrics
            metrics, losses = segmenter.evaluate(pred_masks, gt_masks)

            overall_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    overall_metrics[key] = value.tolist()
                else:
                    overall_metrics[key] = value
            for key, value in losses.items():
                if isinstance(value, np.ndarray):
                    overall_metrics[key] = value.tolist()
                else:
                    overall_metrics[key] = value
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
    7. Extremely important: Do not terminate the conversation until the new preprocessing function is evaluated AND it must be written to the function bank by calling the write_results function.
    8. Recall, this is not a stateful kernel, so all functions, imports, etc. must be provided in the script to be executed.
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
            pipeline_metrics_info=self.pipeline_metrics_info
        )
    
    def run_pipeline_prompt(self) -> str:
        # Create a copy of the pipeline_prompt string
        prompt = self.pipeline_prompt
        
        # Replace placeholders manually
        prompt = prompt.replace("{gpu_id}", str(self.gpu_id))
        prompt = prompt.replace("{seed}", str(self.seed))
        prompt = prompt.replace("{data_path}", str(self.dataset_path))
        prompt = prompt.replace("{function_bank_path}", str(self.function_bank_path))
        prompt = prompt.replace("{output_dir}", '/'.join(self.function_bank_path.split('/')[:-1]))
        
        return prompt
    
    def save_function_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path, output_dir = '/'.join(self.function_bank_path.split('/')[:-1]))
    

class CellposeSegmentationPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell segmentation analysis. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a three-channel image dataset for biological segmentation, consisting of images from different experiments and different settings - a heterogenous dataset of many different object types.  There is a particular focus on biological microscopy images, including cells, sometimes with nuclei labeled in a separate channel.
    The images have pixel values between 0 and 1 and are in float32 format.
    Channel[0] is the nucleus, channel[1] is the cytoplasm, and channel[2] is empty, however not all images have any nuclear data.
    Our goal is to improve the segmentation performance of the neural network by using OpenCV preprocessing functions to improve the quality of the images for downstream segmentation.
    We want to increase the neural network tool's performance at segmenting cells with cell perimeter masks that have high Intersection over Union (IoU) with the ground truth masks.
    The cell images have dimensions (B, L, W, C) = (batch, length, width, channel). To correctly predict masks, the images provided must be in the format of standard ImageData object and must maintain channel dimensions and ordering.   
    """

    task_details = """
    All of you should work together to write {k_word} preprocessing functions to improve segmentation performance using OpenCV functions (APIs provided).
    It might make sense to start the process with small preprocessing functions, and then build up to more complex functions depending on the performance of the previous functions.

    1. Based on previous preprocessing functions and their performance (provided below), suggest {k_word} new preprocessing functions using OpenCV functions (APIs provided below). Successful strategies can include improving upon high performing functions (including tuning the parameters of the function), or exploring the image processing space for novel or different image processing approaches. You can feel free to combine OpenCV functions or suggest novel combinations that can lead to improvements, or modify the parameters of the existing extremely successful functions.
    2. Remember, the images after preprocessing must still conform to the format specified in the ImageData API. Maintenance of channel identity is critical and channels should not be merged.
    3. The environment will handle all data loading, evaluation, and logging of the results.  Your only job is to write the preprocessing function.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    5. Do not terminate the conversation until the new preprocessing functions are evaluated and the numerical performance metrics are logged.
    6. Extremely important: Do not terminate the conversation until each of the {k_word} new preprocessing functions are evaluated AND their results are written to the function bank.
    7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
    8. Do not write any code outside of the preprocessing functions.
    9. Do not modify the masks under any circumstances.  
    10. The preprocessing functions written must return an ImageData object with each image in the batch having the same image resolution (H,W) as the original image.
    """

    pipeline_metrics_info = """
        The following metrics are used to evaluate the performance of the pipeline: average_precision.
        The average_precision is the average precision score of the pipeline at an Intersection over Union (IoU) threshold of 0.5.
        Our ultimate goal is to increase the average_precision as much as possible (0.95 is the target).
        """

    # --- End of CLASS attributes ---

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word, dataset_size, batch_size):
        # Call super using the class attributes
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info, # Access class attribute
            dataset_path=dataset_path,
            # summary_prompt=self.summary_prompt, # Access class attribute
            task_details=self.task_details,     # Access class attribute
            function_bank_path=function_bank_path,
            pipeline_metrics_info=self.pipeline_metrics_info, # Access class attribute
            dataset_size=dataset_size,
            batch_size=batch_size
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path
        self.k = k
        self.k_word = k_word
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def run_pipeline_prompt(self) -> str:
        """
        Reads the template script from a file, replaces configuration
        placeholders, DEDENTS and STRIPS the result, and returns the
        script string containing the function placeholder.
        """
        template_file_path = os.path.join(os.path.dirname(__file__), "cellpose_segmentation_execution-template.py.txt")

        try:
            with open(template_file_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Execution template file not found at: {template_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading execution template file: {e}")

        replacement_values = {
            "gpu_id": str(self.gpu_id),
            "seed": str(self.seed),
            "dataset_path": self.dataset_path.replace("\\", "/"),
            "function_bank_path": self.function_bank_path.replace("\\", "/"),
            "_PREPROCESSING_FUNCTIONS_PLACEHOLDER": _PREPROCESSING_FUNCTION_PLACEHOLDER,
            "sample_k": str(self.k),
            "dataset_size": str(self.dataset_size),
            "batch_size": str(self.batch_size)
        }

        script_with_config = template_content
        for key, value in replacement_values.items():
            placeholder_tag = "{" + key + "}"
            script_with_config = script_with_config.replace(placeholder_tag, value)

        # --- FIX: Apply dedent and strip before returning ---
        dedented_script = textwrap.dedent(script_with_config)
        return dedented_script.strip()
    
    
