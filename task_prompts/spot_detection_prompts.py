from task_prompts.task_prompt import TaskPrompts

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

            spots_data = np.load({data_path}, allow_pickle=True)

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