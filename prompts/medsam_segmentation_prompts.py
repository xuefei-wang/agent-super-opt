from prompts.task_prompts import TaskPrompts, _PREPROCESSING_FUNCTION_PLACEHOLDER
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

    def get_pipeline_metrics_info(self):
        return f"""
    {self.if_advantage("The advantage quantifies how much better this function performs than the expert baseline (if positive) or how much worse than the expert baseline (if negative).")}
    The following metrics are used to evaluate the performance of the pipeline: dsc_metric, nsd_metric.
    - The `dsc_metric` is the dice similarity coefficient (DSC) score of the pipeline and is similar to IoU, measuring the overlap between predicted and ground truth masks.
    - The `nsd_metric` is the normalized surface distance (NSD) score and is more sensitive to distance and boundary calculations.
    """

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, checkpoint_path, k, k_word, advantage_enabled=False, baseline_metric_value=-100):
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
            advantage_enabled=advantage_enabled
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path
        self.checkpoint_path = checkpoint_path
        self.k = k
        self.k_word = k_word
        self.baseline_metric_value = baseline_metric_value
        self.advantage_enabled = advantage_enabled

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
            "_PREPROCESSING_FUNCTIONS_PLACEHOLDER": _PREPROCESSING_FUNCTION_PLACEHOLDER,
            "sample_k": str(self.k),
            "baseline_metric_value": str(self.baseline_metric_value),
            "advantage_enabled": str(self.advantage_enabled),
        }

        script_with_config = template_content
        for key, value in replacement_values.items():
            placeholder_tag = "{" + key + "}"
            script_with_config = script_with_config.replace(placeholder_tag, value)

        # --- FIX: Apply dedent and strip before returning ---
        dedented_script = textwrap.dedent(script_with_config)
        return dedented_script.strip()