from prompts.task_prompts import TaskPrompts
_PREPROCESSING_FUNCTION_PLACEHOLDER = "# --- CODEGEN_PREPROCESSING_FUNCTION_INSERT ---"
import textwrap
import os

class MedSAMSegmentationPrompts(TaskPrompts):
    """Task prompts for cell segmentation analysis."""

    dataset_info = """
        ```markdown
        This is a chest X-ray image lung segmentation dataset. Some images
        feature lungs with tuberculosis, while others are normal. The images have
        dimensions (H, W, C) = (height, width, channel).
        ```
    """
    
    summary_prompt = """
        Summarize the results as a python dictionary, including the newly proposed preprocessing function and its average performance metrics.
        Follow the format:
        {
        "dice_loss": ...,
        "preprocessing_function": "
            ```python
            YOUR_CODE_HERE
            ```
            ",
        }
        """
    
    pipeline_metrics_info = """
        The following metrics are used to evaluate the performance of the pipeline: dice_loss.
        The dice loss is the dice similarity coefficient (DSC) score of the pipeline.
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
            from dotenv import load_dotenv
            from src.data_io import ImageData
            from src.medsam_segmentation import MedSAMTool
            from cv2 import imread
            import os

            gpu_id = {gpu_id}
            checkpoint_path = "{checkpoint_path}"
            seed = {seed}

            # Set up output directory
            output_dir = Path("output")

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

            # Load data
            def _get_binary_masks(nonbinary_mask):
                '''
                Given nonbinary mask which encodes N masks, return N binary masks which
                should encode the same information.
                
                Parameters:
                    - nonbinary_mask: ndarray of shape (H, W)
                Returns:
                    - binary_masks: ndarray of shape (N, H, W)
                '''
                binary_masks = []
                for i in np.unique(nonbinary_mask)[1:]:
                    binary_mask = (nonbinary_mask == i).astype(np.uint8)
                    binary_masks.append(binary_mask.copy())
                binary_masks = np.stack(binary_masks, axis=0)
                return binary_masks
            data_path = '{data_path}'
            img_path = os.path.join(data_path, 'imgs')
            mask_path = os.path.join(data_path, 'gts')

            num_files = 1
            img_files = sorted(glob.glob(os.path.join(img_path, '*')))[:num_files]
            mask_files = sorted(glob.glob(os.path.join(mask_path, '*')))[:num_files]

            raw_images, raw_boxes, raw_masks = [], [], []
            for img_npz_file, mask_npz_file in zip(img_files, mask_files):
                img_data, mask_data = np.load(img_npz_file), np.load(mask_npz_file)  
                
                image, boxes, nonbinary_mask = img_data['imgs'], img_data["boxes"], mask_data['gts']
                binary_masks = _get_binary_masks(nonbinary_mask)
                
                for box, mask in zip(boxes, binary_masks):
                    x1, y1, x2, y2 = box
                    box_string = "[" + ",".join(map(str, [x1, y1, x2, y2])) + "]"
                    
                    raw_images.append(image)
                    raw_boxes.append(box_string)
                    raw_masks.append(mask)

            images = ImageData(raw=raw_images,
                            batch_size=batch_size,
                            image_ids=[i for i in range(len(raw_images))],
                            masks=raw_masks,
                            predicted_masks=raw_masks)

            # TODO: add your preprocessing function here
            # def preprocess_images(images: ImageData) -> ImageData:
            #   YOUR CODE HERE

            images = preprocess_images(images)
            
            # Initialize segmenter
            segmenter = MedSAMTool(gpu_id={gpu_id}, checkpoint_path={checkpoint_path})

            # Run segmenter
            pred_masks = segmenter.predict(images)

            # Calculate Metrics
            losses = segmenter.evaluate(pred_masks, images.masks)

            df = pd.DataFrame([losses])
            overall_losses = df.mean().to_dict()
            logger.info("Overall losses %s", overall_losses)
            ```
    """

    task_details = """
    All of you should work together to write a preprocessing function to improve segmentation performance using OpenCV functions.
    1. Based on previous preprocessing functions and their performance (provided below), suggest a new preprocessing function using OpenCV functions (APIs provided below). Remember, the images after preprocessing must still conform to the format specified in the ImageData API.
    2. Plug the preprocessing function into the pipeline and run the segmenter to calculate the performance metrics, using the provided code snippet. Make sure the loading of images, proprocessing function implementation, and all other pipeline workflow code are contained in the SAME code block.
    3. Save the newly proposed preprocessing function and its performance metrics in the function bank, using the provided script.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.
    6. Do not terminate the conversation until the new preprocessing function is evaluated and the numerical loss metric(s) are logged.
    """


    def __init__(self, gpu_id, checkpoint_path, seed, dataset_path, function_bank_path):
        super().__init__(
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            summary_prompt=self.summary_prompt,
            task_details=self.task_details,
            function_bank_path=function_bank_path,
            pipeline_metrics_info=self.pipeline_metrics_info
        )
    
    def run_pipeline_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path, output_dir = '/'.join(self.function_bank_path.split('/')[:-1]))
    
    def save_function_prompt(self) -> str:
        return self.save_to_function_bank_prompt.format(function_bank_path=self.function_bank_path)