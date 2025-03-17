from dotenv import load_dotenv
import os
import torch
import json
import argparse
import uuid

from autogen import OpenAIWrapper, Cache, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import CodeBlock
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)

from task_prompts.task_prompt import TaskPrompts

from src.utils import set_gpu_device
from src.prompts import (
    sys_prompt_code_writer,
    sys_prompt_code_verifier,
)

# Load environment variables
load_dotenv()

server = LocalJupyterServer()
executor = JupyterCodeExecutor(server, output_dir="output", timeout=300) # very high timeout for long running tasks

def set_up_agents():
    ''' Prepare 3 agents and state transition'''
    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": executor
        }, 
        human_input_mode="NEVER",  # Never take human input for this agent
        # is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    )
    code_writer_agent = ConversableAgent(
        "code_writer",
        system_message=sys_prompt_code_writer,
        llm_config={
            "config_list": [
                {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
    )
    code_verifier_agent = ConversableAgent(
        "code_verifier",
        system_message=sys_prompt_code_verifier,
        llm_config={
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for
        human_input_mode="NEVER",
    )
    
    def state_transition(last_speaker, groupchat):
        ''' Transition between speakers in an agent groupchat '''
        messages = groupchat.messages

        if len(messages) <= 1:
            return code_writer_agent

        if last_speaker is code_writer_agent:
            return code_verifier_agent
        elif last_speaker is code_verifier_agent:
            return code_executor_agent
        elif last_speaker is code_executor_agent:
            if "exitcode: 1" in messages[-1]["content"]:
                return code_writer_agent
            else:
                return code_writer_agent
    
    return code_executor_agent, code_writer_agent, code_verifier_agent, state_transition

# Load documentation and dataset information
# with open("artifacts/docs.md", "r") as file:
#     documentation = file.read()


# Load openCV function APIs
with open("assets/opencv_APIs.txt", "r") as file:
    opencv_APIs = file.read()



def prepare_notes_shared(my_gpu_id):
    notes_shared = f"""
    - Always check the documentation for the available APIs before reinventing the wheel
    - Use GPU {my_gpu_id} for running the pipeline, set `cuda: {my_gpu_id}` in the code snippet!
    - Don't suggest trying larger models as the model size is fixed.
    """
    return notes_shared



notes_pipeline_optimization = f"""
"""


def prepare_prompt_pipeline_optimization(notes_shared, function_bank_path, prompts : TaskPrompts):

    prompt_pipeline_optimization = f"""

    ## About the dataset: 
    {prompts.dataset_info}

    ## Task Details:
    {prompts.task_details}
    
    ## Function bank path:
    {function_bank_path}

    ## OpenCV Function APIs:
    {opencv_APIs}

    ## Additional Notes:
    {notes_shared}
    {notes_pipeline_optimization}

    ## Preprocessing Function API:
    ```python
    from src.data_io import ImageData
    def preprocess_images(images: ImageData) -> ImageData:
        # YOUR CODE HERE
        return preprocessed_images
    ```

    ## Documentation on the `ImageData`:
    ```markdown
    ImageData

    Framework-agnostic container for batched biological image data.
    
    This class provides a standardized structure for storing and managing batched 
    biological image data along with related annotations and predictions.
    All data is stored as numpy arrays for framework independence.

    Arrays are standardized to the following formats:
    - Raw images: (B, H, W, C) where:
        B: batch size
        C: number of channels
        H, W: height and width
    - Masks: (B, 1, H, W) for both ground truth and predictions

    Attributes:
        raw (np.ndarray): Raw image data in (B, H, W, C) format.
        
        batch_size (int): Number of images in the batch.
        
        image_ids (Union[int, str, List[Union[int, str]]]): Unique identifier(s) for images
            in the batch. Can be a single value for batch size 1, or a list matching 
            batch size.
        
        channel_names (Optional[List[str]]): Names of imaging channels in order matching
            raw data channels. Length must equal number of channels.
        
        tissue_types (Optional[Union[str, List[str]]]): Type of biological tissue for each image,
            e.g., ["liver", "kidney"]. Length must equal batch size.
        
        image_mpps (Optional[Union[float, List[float]]]): Microns per pixel resolution for each
            image. Length must equal batch size.
        
        masks (Optional[np.ndarray]): Ground truth segmentation masks in (B, 1, H, W) 
            format. Integer-valued array where 0 is background and positive integers 
            are unique cell identifiers.
        
        cell_types (Optional[List[Dict[int, str]]]): List of mappings from cell 
            identifiers to cell type labels for each image. Length must equal batch size.
        
        predicted_masks (Optional[np.ndarray]): Model-predicted segmentation masks in
            (B, 1, H, W) format.
        
        predicted_cell_types (Optional[List[Dict[int, str]]]): List of mappings from
            cell identifiers to predicted cell types for each image.

    ```
    ## Function for saving the results:
    {prompts.save_function_prompt()}

    ## Code for running segmentation and calculating metrics:
    {prompts.run_pipeline_prompt()}

    
    """

    return prompt_pipeline_optimization


def save_chat_history(chat_history, curr_iter, output_folder):
    
    output_file = os.path.join(output_folder, f"chat_history_ver{curr_iter:03d}.txt")
    with open(output_file, "w") as file:
        for message in chat_history:
            file.write(f"{message['name']}: {message['content']}\n\n")

def save_seed_list(n, file_path):
    '''Saves the seed list to a file path and returns the seed list'''
    uuids = [uuid.uuid4() for _ in range(n)]

    with open(file_path, "w") as file:
        for uid in uuids:
            file.write(f"{uid}\n")

    print(f"Saved {n} seeds to {file_path}")
    
    return uuids
    
def main():
    
    parser = argparse.ArgumentParser(description="SciSeek Agent pipeline")
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to the dataset."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to the output folder."
    )
        
    args = parser.parse_args()
    
    output_function_bank = os.path.join(args.output,"preprocessing_func_bank.json")
    
    # Configuration
    my_gpu_id = 0 # GPU ID to use
    cache_seed = 4 # Cache seed for caching the results
    random_seed = 42 # Random seed for reproducibility
    num_optim_iter = 50 # Number of optimization iterations
    max_round = 100  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10
    
    # Load task prompts
    from task_prompts.spot_detection_prompts import SpotDetectionPrompts
    prompts = SpotDetectionPrompts(gpu_id=0, seed=42, dataset_path=args.dataset, function_bank_path=output_function_bank)

    # Set GPU device
    set_gpu_device(my_gpu_id)
    
    seed_list_file = os.path.join(args.output,"seed_list.txt")
    # Generate seed list
    seed_list = save_seed_list(num_optim_iter, seed_list_file)

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed) as cache:
        
        notes_shared = prepare_notes_shared(my_gpu_id)

        for i in range(num_optim_iter):

            # Set up agents
            code_executor_agent, code_writer_agent, code_verifier_agent, state_transition = set_up_agents()
            
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


            prompt_pipeline_optimization = f"Agent Pipeline Seed {seed_list[i]} \n {prepare_prompt_pipeline_optimization(notes_shared, output_function_bank, prompts)}"
            
            chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_optimization, summary_method="reflection_with_llm",
                                            summary_args={"summary_prompt": prompts.summary_prompt})
            save_chat_history(chat_result.chat_history, i, args.output)

if __name__ == "__main__":
    main()