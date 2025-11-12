from prompts.task_prompts import TaskPrompts, _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER
import textwrap
import os


class SpotDetectionPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell spot detection. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a single-channel cell spot detection dataset. IMPORTANT: The images have dimensions (B, L, W, C) = (batch, length, width, channel).
    The images have pixel values between 0 and 1 and are in float32 format.
    ```
    """

    def get_task_details(self):
        return f"""
    All of you should work together to write {self.k_word} preprocessing and postprocessing function pairs to improve spot detection performance.
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
    The following metrics are used to evaluate the performance of the pipeline: f1_score.
    f1_score: Mean F1 score of predicted spots
    """
    # --- End of CLASS attributes ---

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word, baseline_metric_value=-100):
        # Call super using the class attributes
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info, # Access class attribute
            dataset_path=dataset_path,
            # summary_prompt=None, # Access class attribute
            function_bank_path=function_bank_path,
            k=k,
            k_word=k_word,
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path
        self.k = k
        self.k_word = k_word
        self.baseline_metric_value = baseline_metric_value

    def run_pipeline_prompt(self) -> str:
        """
        Reads the template script from a file, replaces configuration
        placeholders, DEDENTS and STRIPS the result, and returns the
        script string containing the function placeholder.
        """
        template_file_path = os.path.join(os.path.dirname(__file__), "spot_detection_execution-template.py.txt")

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
            "sample_k": str(self.k),
            "baseline_metric_value": str(self.baseline_metric_value),
        }

        script_with_config = template_content
        for key, value in replacement_values.items():
            placeholder_tag = "{" + key + "}"
            script_with_config = script_with_config.replace(placeholder_tag, value)

        # --- FIX: Apply dedent and strip before returning ---
        dedented_script = textwrap.dedent(script_with_config)
        return dedented_script.strip()

    def get_postprocessing_function_api(self):
        api_file_path = os.path.join(os.path.dirname(__file__), "spot_detection_expert_postprocessing_skeleton.py.txt")
        with open(api_file_path, 'r') as f:
            template_content = f.read()

        return textwrap.dedent(template_content)