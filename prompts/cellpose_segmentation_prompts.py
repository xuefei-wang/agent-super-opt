from prompts.task_prompts import TaskPrompts, _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER
import textwrap
import os

class CellposeSegmentationPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell segmentation analysis. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a three-channel image dataset for biological segmentation, consisting of images from different experiments and different settings - a heterogenous dataset of many different object types.  There is a particular focus on biological microscopy images, including cells, sometimes with nuclei labeled in a separate channel.
    The images have pixel values between 0 and 1 and are in float32 format.
    Channel[0] is the nucleus, channel[1] is the cytoplasm, and channel[2] is empty, however not all images have any nuclear data.
    We want to increase the neural network tool's performance at segmenting cells with cell perimeter masks that have high Intersection over Union (IoU) with the ground truth masks.
    The cell images have dimensions (B, L, W, C) = (batch, length, width, channel). To correctly predict masks, the images provided must be in the format of standard ImageData object and must maintain channel dimensions and ordering.   
    """

    def get_task_details(self):
        return f"""
    All of you should work together to write {self.k_word} pairs of preprocessing postprocessing functions that {self.if_advantage("maximize the reported advantages and ")}improve segmentation performance. 
    We provided APIs for both preprocessing and postprocessing functions. You should use functions from useful libraries including but not limited to OpenCV, NumPy, Skimage, Scipy, to implement novel and effective functions.
    It might make sense to start the process with small preprocessing functions, and then build up to more complex functions depending on the performance of the previous functions.

    1. Based on previous preprocessing functions and their performance (provided below), suggest {self.k_word} new unique preprocessing & postprocessing function pairs {self.if_advantage(" that maximize the advantages. Remember, the bigger the advantage for a particular function, the better it performed than average.")}.
    2. For preprocessing, the images after preprocessing must still conform to the format specified in the ImageData API. Maintenance of channel identity is critical and channels should not be merged. For postprocessing, it is also critical to maintain the output format as the sample function provided.
    3. The environment will handle all data loading, evaluation, and logging of the results.  Your only job is to write the preprocessing and postprocessing function pairs.
    4. For this task, if all {self.k_word} function pairs are evaluated correctly, only one iteration is allowed, even if the performance is not satisfactory.
    5. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
    6. Extremely important: Do not terminate the conversation until each of the {self.k_word} new function pairs are evaluated AND their results are written to the function bank.
    7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
    8. Do not write any code outside of the preprocessing and postprocessing functions.
    9. Do not modify the masks under any circumstances.
    """

    def get_pipeline_metrics_info(self):
        return f"""
    {self.if_advantage("The advantage quantifies how much better this function performs than the expert baseline (if positive) or how much worse than the expert baseline (if negative).")}
    The following metrics are used to evaluate the performance of the pipeline: average_precision.
    The average_precision is the average precision score of the pipeline at an Intersection over Union (IoU) threshold of 0.5.
    Our ultimate goal is to {self.if_advantage("maximize the advantage and ")}increase the average_precision as much as possible (0.95 is the target).
    """

    # --- End of CLASS attributes ---

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word, advantage_enabled=False, dataset_size=256, batch_size=16, baseline_metric_value=-100):
        # Call super using the class attributes
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info, # Access class attribute
            dataset_path=dataset_path,
            # summary_prompt=self.summary_prompt, # Access class attribute
            function_bank_path=function_bank_path,
            dataset_size=dataset_size,
            batch_size=batch_size,
            k=k,
            k_word=k_word,
            advantage_enabled=advantage_enabled
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path
        self.k = k
        self.k_word = k_word
        self.baseline_metric_value = baseline_metric_value
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.advantage_enabled = advantage_enabled

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
            "_PREPROCESSING_POSTPROCESSING_FUNCTIONS_PLACEHOLDER": _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER,
            "baseline_metric_value": str(self.baseline_metric_value),
            "advantage_enabled": str(self.advantage_enabled),
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
    
    
    def get_postprocessing_function_api(self):
        api_file_path = os.path.join(os.path.dirname(__file__), "cellpose_segmentation_expert_postprocessing_skeleton.py.txt")
        with open(api_file_path, 'r') as f:
            template_content = f.read()

        return textwrap.dedent(template_content)
    
    def get_postprocessing_function_api_expert(self):
        api_file_path = os.path.join(os.path.dirname(__file__), "cellpose_segmentation_expert_postprocessing.py.txt")
        with open(api_file_path, 'r') as f:
            template_content = f.read()

        return textwrap.dedent(template_content)