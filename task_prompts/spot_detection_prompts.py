from task_prompts.task_prompt import TaskPrompts

class SpotDetectionPrompts(TaskPrompts):
    """Task prompts for biological cell spot detection."""

    dataset_info = """
        ```markdown
        This is a single-channel cell spot detection dataset. The cell images have dimensions (batch, length, width, channel).
        ```
    """

    dataset_path = "/spot_data/SpotNet-v1_1/val.npz"

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
            from src.spot_detection import DeepcellSpotsDetector
            from dotenv import load_dotenv
            from src.data_io import ImageData

            load_dotenv()

            spots_data = np.load('spot_data/SpotNet-v1_1/val.npz', allow_pickle=True)

            images = ImageData(raw = spots_data['X'], batch_size = spots_data['X'].shape[0], image_ids = [i for i in range(spots_data['X'].shape[0])])
            spots_truth = spots_data['y']

            single_image_shape = images.raw.shape[1:3]

            # TODO: add your preprocessing function here
            # def preprocess_images(images: ImageData) -> ImageData:
            #     # YOUR CODE HERE

            images = preprocess_images(images)

            deepcell_spot_detector = DeepcellSpotsDetector()

            # Predict spots
            pred = deepcell_spot_detector.predict(images)

            # Get metrics
            metrics = deepcell_spot_detector.evaluate(single_image_shape, pred, spots_truth)
            ```
    """

    task_details = """
    All of you should work together to write a preprocessing function using OpenCV functions to improve cell spot detection performance.
    1. Based on previous preprocessing functions and their performance (provided below), suggest a new preprocessing function using OpenCV functions (APIs provided below).
    2. Plug the preprocessing function into the pipeline and run the spot detector to calculate the performance metrics, using the provided code snippet.
    3. Save the newly proposed preprocessing function and its performance metrics in the function bank, using the provided script.
    4. The preprocessing function should not change the dimensions of the image.
    5. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    6. Do not terminate the conversation until the new preprocessing function is evaluated.


    """


    def __init__(self, gpu_id, seed):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=self.dataset_path,
            summary_prompt=self.summary_prompt,
            save_to_function_bank_prompt=self.save_to_function_bank_prompt,
            task_details=self.task_details
        )
    
    def run_pipeline_prompt(self) -> str:
        return self.pipeline_prompt.format(gpu_id=self.gpu_id, seed=self.seed)