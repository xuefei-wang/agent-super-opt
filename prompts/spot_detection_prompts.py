from prompts.task_prompts import TaskPrompts
_PREPROCESSING_FUNCTION_PLACEHOLDER = "# --- CODEGEN_PREPROCESSING_FUNCTION_INSERT ---"
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
    }
    """
    # --- End of CLASS attributes ---

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path):
        # Call super using the class attributes
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info, # Access class attribute
            dataset_path=dataset_path,
            summary_prompt=None, # Access class attribute
            task_details=self.task_details,     # Access class attribute
            function_bank_path=function_bank_path,
            pipeline_metrics_info=self.pipeline_metrics_info # Access class attribute
        )
        # Assign instance attributes
        self.gpu_id = gpu_id
        self.seed = seed
        self.dataset_path = dataset_path
        self.function_bank_path = function_bank_path

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
            "_PREPROCESSING_FUNCTION_PLACEHOLDER": _PREPROCESSING_FUNCTION_PLACEHOLDER
        }

        script_with_config = template_content
        for key, value in replacement_values.items():
            placeholder_tag = "{" + key + "}"
            script_with_config = script_with_config.replace(placeholder_tag, value)

        # --- FIX: Apply dedent and strip before returning ---
        dedented_script = textwrap.dedent(script_with_config)
        return dedented_script.strip()