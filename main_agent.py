from dotenv import load_dotenv
import os
import torch
import json

from autogen import OpenAIWrapper, Cache, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import CodeBlock
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)

from src.utils import set_gpu_device
from src.prompts import (
    sys_prompt_code_writer,
    sys_prompt_code_verifier,
)

# Load environment variables
load_dotenv()


def set_up_agents(max_round):
    server = LocalJupyterServer()
    executor = JupyterCodeExecutor(server, output_dir="output", timeout=10000) # very high timeout for long running tasks

    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,
        code_execution_config={
            "executor": executor
        },  # Use the docker command line code executor
        # human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
        human_input_mode="NEVER",  # Never take human input for this agent
        # is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    )

    code_writer_agent = ConversableAgent(
        "code_writer",
        system_message=sys_prompt_code_writer,
        llm_config={
            # "config_list": [{"model": "gemini-1.5-pro", "api_key": os.environ["GEMINI_API_KEY"], "api_type": "google"}],
            "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}],
        },
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    code_verifier_agent = ConversableAgent(
        "code_verifier",
        system_message=sys_prompt_code_verifier,
        llm_config={
            # "config_list": [{"model": "gemini-1.5-pro", "api_key": os.environ["GEMINI_API_KEY"], "api_type": "google"}],
            "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}],
        },
        code_execution_config=False,
        human_input_mode="NEVER",
    )


    def state_transition(last_speaker, groupchat):
        """Determine the next speaker in the group chat.

        Args:
            last_speaker: The previous speaker in the conversation
            groupchat: The group chat instance

        Returns:
            ConversableAgent: The next speaker, or None to terminate
        """
        messages = groupchat.messages

        if len(messages) <= 1:  # First round
            return code_writer_agent

        if "TERMINATE" in messages[-1]["content"]: # Terminate if the last message contains "TERMINATE"
            return None

        if last_speaker is code_writer_agent:
            return code_verifier_agent
        elif last_speaker is code_verifier_agent:
            return code_executor_agent
        elif last_speaker is code_executor_agent:
            return code_writer_agent
        
    # Set up group chat
    group_chat = GroupChat(
        agents=[
            code_executor_agent,
            code_writer_agent,
            code_verifier_agent,
        ],
        messages=[],
        max_round=max_round,
        send_introductions=True,
        speaker_selection_method=state_transition,
    )

    # Initialize group chat manager
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={
            # "config_list": [{"model": "gemini-1.5-pro", "api_key": os.environ["GEMINI_API_KEY"], "api_type": "google"}],
            "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}],
        },
        is_termination_msg=lambda msg: (
            "TERMINATE" in msg["content"] if msg["content"] else False
        ),
    )

    return code_executor_agent, group_chat_manager

# Load documentation and dataset information
with open("artifacts/docs.md", "r") as file:
    documentation = file.read()

# Load function bank
with open("output/preprocessing_func_bank.json", "r") as file:
    function_bank = json.load(file)

# Load openCV function APIs
with open("assets/opencv_APIs.md", "r") as file:
    opencv_APIs = file.read()


dataset_info = """
```markdown
This is a single-channel nuclear segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset covering
five different cell lines (NIH-3T3, HeLa-S3, HEK293, RAW 264.7, and PC-3).
```
"""

dataset_path = "/data/user-data/xwang3/DynamicNuclearNet/DynamicNuclearNet-segmentation-v1_0/val.npz"

def prepare_notes_shared(my_gpu_id):
    notes_shared = f"""
    - Always check the documentation for the available APIs before reinventing the wheel
    - Use GPU {my_gpu_id} for running the pipeline, set `cuda: {my_gpu_id}` in the code snippet!
    - Don't suggest trying larger models as the model size is fixed.
    """
    return notes_shared



notes_pipeline_optimization = f"""
"""


summary_prompt = """
Summarize the results as a python dictionary, including the newly proposed preprocessing function and its average performance metrics.
Follow the format:
{
    "mean_iou": ...,
    "precision": ...,
    "recall": ...,
    "f1_score": ...,
    "preprocessing_function": "
        ```python
        YOUR_CODE_HERE
        ```
        ",
}
"""


def prepare_prompt_pipeline_optimization(notes_shared):

    prompt_pipeline_optimization = f"""
    # Cell Segmentation Analysis Pipeline Optimization
    ## Objective:
    Optimize the pipeline for cell segmentation analysis by suggesting new preprocessing functions.

    ## About the dataset: 
    {dataset_info}

    ## Task Details:
    All of you should work together to write a preprocessing function to improve segmentation performance using OpenCV functions.
    1. Based on previous preprocessing functions and their performance (provided below), suggest a new preprocessing function using OpenCV functions (APIs provided below).
    2. Plug the preprocessing function into the pipeline and run the segmenter to calculate the performance metrics, using the provided code snippet.
    3. Save the newly proposed preprocessing function and its performance metrics in the function bank, using the provided script.
    4. Only one iteration is allowed for this task, even if the performance is not satisfactory.

    ## Previous preprocessing functions and their performance (might be empty):
    {function_bank}

    ## OpenCV Function APIs:
    {opencv_APIs}

    ## Additional Notes:
    {notes_shared}
    {notes_pipeline_optimization}

    ## Preprocessing Function API:
    ```python
    from src.data_io import ImageData
    def preprocess_image(image: ImageData) -> Image:
        # YOUR CODE HERE
        return image
    ```

    ## Documentation on the `ImageData`:
    ```markdown
    ImageData

    Container for biological image data and associated metadata.

    This class provides a standardized structure for storing and managing biological
    image data along with related annotations and predictions. It handles data
    validation and format standardization automatically.

    The class standardizes array shapes as follows:
    - Raw images: (C, H, W) format where C is number of channels
    - Masks: (1, H, W) format for both ground truth and predictions

    Attributes:
        raw (np.ndarray): Raw image data in (C, H, W) format. For multichannel data,
            C corresponds to different imaging channels (e.g., fluorescence markers).
            For single-channel data, C=1.
        
        image_id (Union[int, str]): Unique identifier for the image. Can be numeric
            index or string identifier.
        
        channel_names (Optional[List[str]]): Names of imaging channels in order matching
            raw data channels. For multichannel data, must have length equal to
            number of channels. For single-channel data, defaults to ["channel_0"].
        
        tissue_type (Optional[str]): Type of biological tissue in the image, e.g.,
            "liver", "kidney", etc.
        
        cell_types (Optional[List[str]]): List of cell types expected to be present
            in the image. Used for reference and verification.
        
        image_mpp (Optional[float]): Microns per pixel resolution of the image.
            Used for size-based analysis and visualization scaling.
        
        mask (Optional[np.ndarray]): Ground truth segmentation mask in (1, H, W) format.
            Integer-valued array where 0 is background and positive integers are
            unique cell identifiers.
        
        cell_type_info (Optional[Dict[int, str]]): Mapping from cell identifiers in
            the ground truth mask to their corresponding cell type labels.
        
        predicted_mask (Optional[np.ndarray]): Model-predicted segmentation mask in
            (1, H, W) format. Integer-valued array where 0 is background and
            positive integers are unique cell identifiers.
        
        predicted_cell_types (Optional[Dict[int, str]]): Mapping from cell identifiers
            in the predicted mask to their predicted cell type labels.

    Example:
        >>> image_data = ImageData(
        ...     raw=np.array(...),  # Shape (2, 512, 512)
        ...     image_id="sample_001",
        ...     channel_names=["DAPI", "CD3"],
        ...     tissue_type="lymph_node",
        ...     mask=np.array(...)  # Shape (1, 512, 512)
        ... )

    ```
    ## Function for saving the results:
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

    ## Code for running segmentation and calculating metrics:
    ```python
    import numpy as np
    import tensorflow as tf
    from src.data_io import NpzDataset
    from src.segmentation import MesmerSegmenter, calculate_metrics
    import logging
    import pandas as pd
    from pathlib import Path

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()  # This keeps console logging
        ]
    )
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    data_path = "/data/user-data/xwang3/DynamicNuclearNet/DynamicNuclearNet-segmentation-v1_0/test.npz"
    dataset = NpzDataset(data_path)
    indices = np.random.choice(len(dataset), 10)
    images = dataset.load(indices)

    # Initialize segmenter
    segmenter = MesmerSegmenter()

    # Process each image
    metrics_list = []
    for image in images:
        logger.info(f"Processing image",  image.image_id)

        try:
            # TODO: add your preprocessing function here
            # image = preprocess_image(image)

            segmented_image = segmenter.predict(image)
            metrics = calculate_metrics(
                segmented_image.mask, segmented_image.predicted_mask
            )
            logger.info(f"Metrics: ", metrics)
            metrics_list.append(metrics)
        
        except Exception as e:
            logger.error(f"Segmentation failed for image", image.image_id, str(e))
            continue

    df = pd.DataFrame(metrics_list)
    avg_metrics = df.mean().to_dict()
    logger.info(f"Average Metrics: ", avg_metrics)
    ```

    """

    return prompt_pipeline_optimization


def save_chat_history(chat_history, curr_iter):
    with open(f"output/chat_history_ver{curr_iter:03d}.txt", "w") as file:
        for message in chat_history:
            file.write(f"{message['name']}: {message['content']}\n\n")

def main():
    # Configuration
    my_gpu_id = 7 # GPU ID to use
    cache_seed = 4 # Cache seed for caching the results
    num_optim_iter = 5 # Number of optimization iterations
    max_round = 100  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10

    # Set GPU device
    set_gpu_device(my_gpu_id)

    # Set up agents
    code_executor_agent, group_chat_manager = set_up_agents(max_round=max_round)

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed) as cache:
        
        notes_shared = prepare_notes_shared(my_gpu_id)

        for i in range(0, num_optim_iter+1):
            prompt_pipeline_optimization = prepare_prompt_pipeline_optimization(notes_shared)
            
            chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_optimization, summary_method="reflection_with_llm",
                                            summary_args={"summary_prompt": summary_prompt})
            save_chat_history(chat_result.chat_history, i)

if __name__ == "__main__":
    main()