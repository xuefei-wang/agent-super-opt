from prompts.task_prompts import TaskPrompts, _PREPROCESSING_FUNCTION_PLACEHOLDER
import textwrap
import os

class SpotDetectionPrompts(TaskPrompts):
    """Task prompts for biological cell spot detection."""

    dataset_info = """
        ```markdown
        This is a single-channel cell spot detection dataset. IMPORTANT: The cell images have dimensions (B, L, W, C) = (batch, length, width, channel).
        ```
    """
    
    summary_prompt = """
        Summarize the results as a python dictionary, including the newly proposed preprocessing function and its average performance metrics.
        Follow the format:
        {
        "class_loss": ...,
        "regress_loss": ...,
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
            
            with open('{function_bank_path}', 'r') as file:
                json_array = json.load(file)

            with open('{function_bank_path}', 'w') as file:
                json_data = metrics_dict
                json_data["preprocessing_function"] = inspect.getsource(preprocessing_fn)
                json_array.append(json_data)
                json.dump(json_array, file)
            
            print("Finished writing preprocessing function to function bank")
        ```
    """


    pipeline_prompt = """
        ```python
            import numpy as np
            from src.spot_detection import DeepcellSpotsDetector
            from dotenv import load_dotenv
            from src.data_io import ImageData

            load_dotenv()

            spots_data = np.load("{data_path}", allow_pickle=True)

            images = ImageData(raw = spots_data['X'], batch_size = spots_data['X'].shape[0], image_ids = [i for i in range(spots_data['X'].shape[0])])
            spots_truth = spots_data['y']

            # TODO: add your preprocessing function here
            # def preprocess_images(images: ImageData) -> ImageData:
            #     # YOUR CODE HERE

            images = preprocess_images(images)

            deepcell_spot_detector = DeepcellSpotsDetector()

            # Predict spots
            pred = deepcell_spot_detector.predict(images)

            # Get metrics
            metrics = deepcell_spot_detector.evaluate(pred, spots_truth)
            ```
    """

    task_details = """
    You will work together to complete the following instructions in order:
    1. View the function bank provided in the prompt to see previous preprocessing functions and their performance metrics.
    2. Based on previous evaluations, suggest a new unique preprocessing function that may improve the performance metrics of the spot detector.
    3. Plug the preprocessing function into the pipeline and run the spot detector to calculate the performance metrics, using the provided code snippet.
    4. Save the newly proposed preprocessing function and its performance metrics in the function bank, using the provided script. Do not terminate until you can verify the output of the code. 
    Make sure that the entire pipeline runs to end-to-end with the new preprocessing function and computes metrics before saving to function bank.

    """

    pipeline_metrics_info = """
    {
    class_loss: loss from one-hot encoded 2D matrix, where 1 is a spot and 0 is not a spot
    regress_loss: loss 2D matrix where each entry is distance from a predicted spot
    f1_score: Mean F1 score of predicted spots
    }
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
        return self.pipeline_prompt.format(gpu_id=self.gpu_id, seed=self.seed, data_path=self.dataset_path)
    
    def save_function_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path)


class SpotDetectionPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell spot detection. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a single-channel cell spot detection dataset. IMPORTANT: The cell images have dimensions (B, L, W, C) = (batch, length, width, channel).
    ```
    """

    def get_task_details(self):
        return f"""
    All of you should work together to write {self.k_word} preprocessing functions to {self.if_advantage("maximize the reported advantages and ")}improve spot detection performance using OpenCV functions.
    1. Based on previous preprocessing functions and their performance (provided below), suggest {self.k_word} new unique preprocessing functions using OpenCV functions (APIs provided below){self.if_advantage(" that maximize the advantages. Remember, the bigger the advantage for a particular function, the better it performed than average")}.
    2. The environment will handle all data loading, evaluation, and logging of the results. Your only job is to write the preprocessing functions.
    3. Do not terminate the conversation until the new preprocessing functions are evaluated and the numerical performance metrics are logged.
    4. For this task, if all {self.k_word} functions are evaluated correctly, only one iteration is allowed, even if the performance is not satisfactory.
    5. Do not terminate the conversation until the new preprocessing functions are evaluated and the numerical performance metrics are logged.
    6. Extremely important: Do not terminate the conversation until each of the {self.k_word} new preprocessing functions are evaluated AND their results are written to the function bank.
    7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
    8. Do not write any code outside of the preprocessing functions.
    """

    def get_pipeline_metrics_info(self):
        return f"""
    {{
    {self.if_advantage("advantage: score which quantifies how much better this function performs than the expert baseline (if positive) or how much worse than the expert baseline (if negative)")}
    class_loss: loss from one-hot encoded 2D matrix, where 1 is a spot and 0 is not a spot
    regress_loss: loss 2D matrix where each entry is distance from a predicted spot
    f1_score: Mean F1 score of predicted spots
    }}
    """
    # --- End of CLASS attributes ---

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word, advantage_enabled=False, baseline_metric_value=-100):
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
        self.advantage_enabled = advantage_enabled

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