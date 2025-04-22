from prompts.task_prompts import TaskPrompts

class SpotDetectionPrompts(TaskPrompts):
    """Task prompts for biological cell spot detection."""

    dataset_info = """
        ```markdown
        This is a single-channel cell spot detection dataset. IMPORTANT: The cell images have dimensions (B, L, W, C) = (batch, length, width, channel).
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
            
            # TODO: fill in your {k_word} preprocessing functions below
            {functions}
            
            preprocessing_fns = [{function_names}]
            
            metrics = []
            
            deepcell_spot_detector = DeepcellSpotsDetector()
            
            for preprocessing_fn in preprocessing_fns:
            
                # Apply preprocessing function to images
                images = preprocessing_fn(images)
            
                # Predict spots
                pred = deepcell_spot_detector.predict(images)
            
                # Add individual metrics
                metrics.append(deepcell_spot_detector.evaluate(pred, spots_truth))
                
            compute_reward = lambda m: m["f1_score"]
            
            group_baseline = np.mean(list(map(compute_reward, metrics)))
            
            metrics = [{{**m, "advantage": compute_reward(m) - group_baseline}} for m in metrics]
            ```
    """

    task_details = """
    You will work together to complete the following instructions in order:
    1. View the functions from the function bank provided in the prompt to see previous preprocessing functions, their performance metrics, and their advantages.
    2. Based on previous evaluations and their respective advantages (provided below), suggest {k_word} new unique preprocessing functions that maximize the advantages and may improve the performance metrics of the spot detector. Remember, the bigger the advantage for a particular function, the better it performed than average.
    3. Plug each of the {k_word} preprocessing functions into the pipeline and run the spot detector to calculate the performance metrics and advantages of each one, using the provided code snippet.
    4. Save the {k_word} newly proposed preprocessing functions, their performance metrics, and their advantages in the function bank, using the provided script. Do not terminate until you can verify the output of the code. 
    Make sure that the entire pipeline runs to end-to-end with the new preprocessing functions and computes metrics before saving to function bank.

    """

    pipeline_metrics_info = """
    {
    advantage: score which quantifies how much better this function performs than average (if positive) or how much worse than average (if negative)
    f1_score: mean f1 score of predicted spots
    class_loss: loss from one-hot encoded 2D matrix, where 1 is a spot and 0 is not a spot
    regress_loss: loss 2D matrix where each entry is distance from a predicted spot
    }
    """


    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, grpo_k, grpo_k_word):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            summary_prompt=None,
            pipeline_prompt=self.pipeline_prompt,
            task_details=self.task_details,
            function_bank_path=function_bank_path,
            pipeline_metrics_info=self.pipeline_metrics_info,
            grpo_k=grpo_k,
            grpo_k_word=grpo_k_word,
        )

    def run_pipeline_prompt(self) -> str:
        return super().run_pipeline_prompt()

    def save_function_prompt(self) -> str:
        return super().save_function_prompt()

    def task_details_prompt(self) -> str:
        return super().task_details_prompt()