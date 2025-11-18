from prompts.task_prompts import TaskPrompts, _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER
import textwrap
import os

class MedSAMSegmentationPromptsWithSkeleton(TaskPrompts):
    """Task prompts for MedSAM segmentation analysis. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown

    This is large-scale medical image segmentation dataset covering the 
    dermoscopy/xray modality. The images have dimensions (H, W, C) = (height, width, channel).
    ```
    """

    def get_task_details(self):
        return  f"""
        All of you should work together to write {self.k_word} preprocessing and postprocessing function pairs to improve segmentation performance.
        We provided APIs for both preprocessing and postprocessing functions. You should use functions from useful libraries including but not limited to OpenCV, NumPy, Skimage, Scipy, to implement novel and effective functions.
        1. Based on previous preprocessing and postprocessing functions and their performance (provided below), suggest {self.k_word} new unique function pairs using.
        2. The environment will handle all data loading, evaluation, and logging of the results. Your only job is to write the preprocessing and postprocessing functions.
        3. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
        4. For this task, if all {self.k_word} functions are evaluated correctly, only one iteration is allowed, even if the performance is not satisfactory.
        5. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
        6. Extremely important: Do not terminate the conversation until each of the {self.k_word} new function pairs are evaluated AND their results are written to the function bank.
        7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
        8. Do not write any code outside of the preprocessing and postprocessing functions.
        9. For preprocessing, the images after preprocessing must still conform to the format specified in the ImageData API. Maintenance of channel identity is critical and channels should not be merged. For postprocessing, it is also critical to maintain the output format as the sample function provided.
        """

    def get_pipeline_metrics_info(self):
        return f"""
    The following metrics are used to evaluate the performance of the pipeline: dsc_metric, nsd_metric.
    - The `dsc_metric` is the dice similarity coefficient (DSC) score of the pipeline and is similar to IoU, measuring the overlap between predicted and ground truth masks.
    - The `nsd_metric` is the normalized surface distance (NSD) score and is more sensitive to distance and boundary calculations.
    """

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, checkpoint_path, k, k_word):
        # Call super using the class attributes
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info, # Access class attribute
            dataset_path=dataset_path,
            function_bank_path=function_bank_path,
            checkpoint_path=checkpoint_path,
            k=k,
            k_word=k_word,
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path
        self.checkpoint_path = checkpoint_path
        self.k = k
        self.k_word = k_word

    def run_pipeline_prompt(self) -> str:
        """
        Reads the template script from a file, replaces configuration
        placeholders, DEDENTS and STRIPS the result, and returns the
        script string containing the function placeholder.
        """
        template_file_path = os.path.join(os.path.dirname(__file__), "medsam_segmentation_execution-template.py.txt")

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
            "checkpoint_path": self.checkpoint_path.replace("\\", "/"),
            "_PREPROCESSING_POSTPROCESSING_FUNCTIONS_PLACEHOLDER": _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER,
            "sample_k": str(self.k),
        }

        script_with_config = template_content
        for key, value in replacement_values.items():
            placeholder_tag = "{" + key + "}"
            script_with_config = script_with_config.replace(placeholder_tag, value)

        # --- FIX: Apply dedent and strip before returning ---
        dedented_script = textwrap.dedent(script_with_config)
        return dedented_script.strip()
    
    def get_postprocessing_function_api(self):
        api_file_path = os.path.join(os.path.dirname(__file__), "medsam_segmentation_expert_postprocessing_skeleton.py.txt")
        with open(api_file_path, 'r') as f:
            template_content = f.read()

        return textwrap.dedent(template_content)