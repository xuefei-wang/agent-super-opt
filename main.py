from dotenv import load_dotenv
import os
import torch
import json
import argparse
import uuid

from autogen import OpenAIWrapper, Cache, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import CodeBlock, CodeExecutor, DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)

from prompts.task_prompts import TaskPrompts

from src.utils import set_gpu_device
from prompts.agent_prompts import (
    sys_prompt_code_writer,
    sys_prompt_code_writer_commandline,
    sys_prompt_code_verifier,
)

from utils.function_bank_utils import top_n, last_n, pretty_print_list, worst_n

from utils.exploration_utils import ProbabilisticExploration

# Load environment variables
load_dotenv()


def set_up_agents(executor: CodeExecutor):
    ''' Prepare 3 agents and state transition'''
    if isinstance(executor, LocalCommandLineCodeExecutor):
        code_writer_prompt = sys_prompt_code_writer_commandline
    elif isinstance(executor, JupyterCodeExecutor):
        code_writer_prompt = sys_prompt_code_writer
    else:
        raise ValueError(f"Executor type {type(executor)} not supported")
    
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
        system_message=code_writer_prompt,
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
    - Be sure to utilize to_numpy() in the ImageData class if errors with lists occurs #TODO: DELETE
    - THE PROVIDED EVALUATION PIPELINE WORKS OUT OF THE BOX, IF THERE IS AN ERROR IT IS WITH THE PREPROCESSING FUNCTION

"""


def prepare_prompt_pipeline_optimization(notes_shared, function_bank_path, prompts : TaskPrompts, mode_prompt: str):

    prompt_pipeline_optimization = f"""

    ## About the dataset: 
    {prompts.dataset_info}

    ## Task Details:
    {prompts.task_details}

    ## Task Metrics Details:
    {prompts.pipeline_metrics_info}
    
    ## Top 3 best performing functions from function bank:
    {pretty_print_list(top_n(function_bank_path, n = 3, sorting_function=lambda x: x["class_loss"]))}

    ## Worst 3 performing functions from the function bank:
    {pretty_print_list(worst_n(function_bank_path, n = 3, sorting_function=lambda x: x["class_loss"]))}

    ## Execution history / most recent 3 functions from function bank:
    {pretty_print_list(last_n(function_bank_path, n = 3))}

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

    ## Documentation on the `ImageData` class:
    ```markdown
    Framework-agnostic container for batched image data. Handles variable
    image resolutions
    
    This class provides a standardized structure for storing and managing batched 
    image data along with related annotations and predictions.
    Data is internally converted to lists of arrays for flexibility with varying image sizes.

    The class accepts both lists of arrays and numpy arrays as input, but will convert them
    internally to lists to support variable-sized images across different frameworks.

    Attributes:
        raw (Union[List[np.ndarray], np.ndarray]): Raw image data, can be provided as either 
            a list of arrays or a numpy array. Each image should have shape (H, W, C).
        
        batch_size (Optional[int]): Number of images to include in the batch. Can be smaller 
            than the total dataset size. If None, will use the full dataset size.
        
        image_ids (Union[List[int], List[str], None]): Unique identifier(s) for images
            in the batch as a list. If None, auto-generated integer IDs [0,1,2,...] will be created.
        
        channel_names (Optional[List[str]]): Names of imaging channels in order matching
            raw data channels. Length must equal number of channels.
        
        masks (Optional[Union[List[np.ndarray], np.ndarray]]): Ground truth segmentation masks.
            Integer-valued arrays where 0 is background and positive integers are unique 
            object identifiers. Each mask should have shape (H, W, 1) or (H, W).
        
        predicted_masks (Optional[Union[List[np.ndarray], np.ndarray]]): Model-predicted 
            segmentation masks. Each mask should have shape (H, W, 1) or (H, W).
        
        predicted_classes (Optional[List[Dict[int, str]]]): List of mappings from
            object identifiers to predicted classes for each image.

    Functions:
        to_numpy (self) -> 'ImageDataNP': Converts ImageData to ImageDataNP, which has the same API but uses numpy arrays internally.
            Better suited for datasets using numpy arrays.

    ```
    ## Function for saving the results:
    {prompts.save_function_prompt()}

    ## Code for running pipeline and calculating metrics:
    {prompts.run_pipeline_prompt()}


    ## Priority:
    {mode_prompt}

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
    
def main(args: argparse.Namespace, executor: CodeExecutor):
    
    output_function_bank = os.path.join(args.output,"preprocessing_func_bank.json")
    
    # Configuration
    my_gpu_id = args.gpu_id # GPU ID to use
    cache_seed = 4 # Cache seed for caching the results
    random_seed = args.random_seed # Random seed for reproducibility
    num_optim_iter = 50 # Number of optimization iterations
    max_round = 100  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10
    checkpoint_path = args.checkpoint_path
    
    # Load task prompts
    if args.experiment_name == "spot_detection":
        from prompts.spot_detection_prompts import SpotDetectionPrompts
        prompt_class = SpotDetectionPrompts
        prompts = prompt_class(gpu_id=args.gpu_id, seed=args.random_seed, dataset_path=args.dataset, function_bank_path=output_function_bank)
    elif args.experiment_name == "cellpose_segmentation":
        from prompts.cellpose_segmentation_prompts import CellposeSegmentationPrompts
        prompt_class = CellposeSegmentationPrompts
        prompts = prompt_class(gpu_id=args.gpu_id, seed=args.random_seed, dataset_path=args.dataset, function_bank_path=output_function_bank)
    elif args.experiment_name == "medSAM_segmentation":
        from prompts.medsam_segmentation_prompts import MedSAMSegmentationPrompts
        prompt_class = MedSAMSegmentationPrompts
        prompts = prompt_class(gpu_id=args.gpu_id, seed=args.random_seed, dataset_path=args.dataset, function_bank_path=output_function_bank, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Experiment name {args.experiment_name} not supported")

    # Set GPU device
    set_gpu_device(my_gpu_id)
    
    seed_list_file = os.path.join(args.output,"seed_list.txt")
    # Generate seed list
    seed_list = save_seed_list(num_optim_iter, seed_list_file)

    # Load mode prompts
    exploration_modes = ProbabilisticExploration(
        seed=random_seed,
        temperature=0.5
    )

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed, cache_path_root=f"{args.output}/cache") as cache:
        
        notes_shared = prepare_notes_shared(my_gpu_id)

        for i in range(num_optim_iter):

            # Set up agents
            code_executor_agent, code_writer_agent, code_verifier_agent, state_transition = set_up_agents(executor)
            
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
                                            summary_args={"summary_prompt": prompts.summary_prompt},
                                            cache=cache)
            save_chat_history(chat_result.chat_history, i, args.output)

if __name__ == "__main__":

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
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=["spot_detection", "cellpose_segmentation", "medSAM_segmentation"],
        help="Name of the experiment. Must be one of: spot_detection, cellpose_segmentation, medSAM_segmentation"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the model checkpoint file. Only used for medSAM segmentation."
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use."
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        required=False,
        help="Random seed to use."
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="The working directory for the agent to access source code."
    )
    
    args = parser.parse_args()

    # server = LocalJupyterServer(log_file=os.path.join("..", args.output, "jupyter_gateway.log"))
    # server = LocalJupyterServer(log_file=None)
    # executor = JupyterCodeExecutor(server, output_dir=args.output, timeout=300) # very high timeout for long running tasks
    executor = LocalCommandLineCodeExecutor(work_dir=args.work_dir, timeout=300)
        
    main(args, executor) 