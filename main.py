from dotenv import load_dotenv
import os
import torch
import json
import argparse
import uuid
import random
from datetime import datetime
import os

from autogen import OpenAIWrapper, Cache, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import CodeBlock, CodeExecutor, DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from utils.executors import TemplatedLocalCommandLineCodeExecutor
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)

from prompts.task_prompts import TaskPrompts

from src.utils import set_gpu_device
from prompts.agent_prompts import (
    sys_prompt_code_writer,
    # sys_prompt_code_verifier,
)

from utils.function_bank_utils import top_n, last_n, pretty_print_list, worst_n

# Load environment variables
load_dotenv()


def set_up_agents(executor: CodeExecutor, llm_model: str, k, k_word):
    ''' Prepare 3 agents and state transition'''
    if isinstance(executor, JupyterCodeExecutor) or isinstance(executor, LocalCommandLineCodeExecutor) or isinstance(executor, TemplatedLocalCommandLineCodeExecutor):
        code_writer_prompt = sys_prompt_code_writer(k, k_word)
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
                # {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}
                {"model": llm_model, "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
    )
    # code_verifier_agent = ConversableAgent(
    #     "code_verifier",
    #     system_message=sys_prompt_code_verifier,
    #     llm_config={
    #         "config_list": [
    #             {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
    #         ]
    #     },
    #     code_execution_config=False,  # Turn off code execution for
    #     human_input_mode="NEVER",
    # )
    
    def state_transition(last_speaker, groupchat):
        ''' Transition between speakers in an agent groupchat '''
        messages = groupchat.messages

        if len(messages) <= 1:
            return code_writer_agent

        if last_speaker is code_writer_agent:
            return code_executor_agent #code_verifier_agent
        # elif last_speaker is code_verifier_agent:
        #     return code_executor_agent
        elif last_speaker is code_executor_agent:
            if "exitcode: 1" in messages[-1]["content"]:
                return code_writer_agent
            else:
                return code_writer_agent
    
    # return code_executor_agent, code_writer_agent, code_verifier_agent, state_transition
    return code_executor_agent, code_writer_agent, state_transition


# Load documentation and dataset information
# with open("artifacts/docs.md", "r") as file:
#     documentation = file.read()


# Load openCV function APIs
with open("assets/opencv_APIs.txt", "r") as file:
    opencv_APIs = file.read()



def prepare_notes_shared(my_gpu_id, max_rounds):
    notes_shared = f"""
    - Always check the documentation for the available APIs before reinventing the wheel
    - Use GPU {my_gpu_id} for running the pipeline, set `cuda: {my_gpu_id}` in the code snippet!
    - You only have {max_rounds} rounds of each conversation to optimize the preprocessing function.
    - Don't suggest trying larger models as the model size is fixed.
    """
    return notes_shared



notes_pipeline_optimization = f"""
    - THE PROVIDED EVALUATION PIPELINE WORKS OUT OF THE BOX, IF THERE IS AN ERROR IT IS WITH THE PREPROCESSING FUNCTION

"""

def function_bank_sample(function_bank_path: str, n_top: int, n_worst: int, n_last: int, sorting_function: callable, current_iteration: int, history_threshold: int=0, total_iterations: int=30, maximize = True):
    ''' Returns a sample of the function bank 
    
    Args:
        function_bank_path (str): Path to the function bank
        n_top (int): Number of top performing functions to return
        n_worst (int): Number of worst performing functions to return
        n_last (int): Number of last executed functions to return
        sorting_function (callable): Function to sort the function bank
        current_iteration (int): Current iteration of the pipeline
        history_threshold (int): Threshold for showing the function bank history
        total_iterations (int): Total number of iterations for the pipeline
        maximize (bool): Whether to maximize or minimize the metric
    Returns:
        str: Inlined samples of the function bank
    
    
    Example:
    ```python
    function_bank_sample(function_bank_path, n_top=100, n_worst=0, n_last=0, sorting_function=lambda x: x["F1"])
    ```
    Usage: Return all elements of the function bank, set n_top to a high number
    and n_worst to 0, and n_last to 0.
    
    """
    
    '''
    
    if current_iteration < history_threshold:
        return f"Function bank history will be shown after iteration {history_threshold}, you are currently on iteration {current_iteration} of {total_iterations}"

    sample = ""

    if(n_top > 0):
        sample += f"""
    ## Top {n_top} performing functions from function bank:
    {pretty_print_list(top_n(function_bank_path, n = n_top, sorting_function=sorting_function, maximize=maximize))}

        """
    
    if(n_worst > 0):
        sample += f"""
    ## Worst {n_worst} performing functions from the function bank:
    {pretty_print_list(worst_n(function_bank_path, n = n_worst, sorting_function=sorting_function, maximize=maximize))}

        """

    if(n_last > 0):
        sample += f"""
    ## Execution history / most recent {n_last} functions from function bank:
    {pretty_print_list(last_n(function_bank_path, n = n_last))}

        """

    return sample

def prepare_prompt_pipeline_optimization(
        notes_shared: str, 
        function_bank_path: str, 
        prompts : TaskPrompts, 
        sampling_function: callable, 
        current_iteration: int, 
        history_threshold: int=0, 
        total_iterations: int=30, 
        maximize = True, 
        n_top: int=5,
        n_worst: int=5, 
        n_last: int=5,
        baseline_metric: str = ""):

    prompt_pipeline_optimization = f"""

    
    ## Preprocessing Functions API:
    ```python
    # Necessary imports for any function's logic (if any)
    # Do not import ImageData in the functions, it is already imported in the environment
    # All preprocessing function names should be of the form preprocess_images_i where i enumerates the preprocessing function, beginning at 1
    import cv2 as cv
    def preprocess_images_i(images: ImageData) -> ImageData:
        # Function logic here
        processed_images_list = []
        for img_array in images.raw:
            img_array = np.copy(img_array) # Make a copy of the image array to avoid modifying the original
            processed_img = img_array # Replace with actual processing
            processed_images_list.append(processed_img)
        output_data = ImageData(raw=processed_images_list, batch_size=images.batch_size)
        return output_data
    ```
    ## About the dataset: 
    {prompts.dataset_info}
    {baseline_metric}

    ## Task Details:
    {prompts.task_details}

    ## Task Metrics Details:
    {prompts.pipeline_metrics_info}
    
    ## Function bank sample:
    {function_bank_sample(function_bank_path, n_top=n_top, n_worst=n_worst, n_last=n_last, sorting_function=sampling_function, current_iteration=current_iteration, history_threshold=history_threshold, total_iterations=total_iterations)}

    ## OpenCV Function APIs:
    {opencv_APIs}

    ## Additional Notes:
    {notes_shared}
    {notes_pipeline_optimization}


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
    """

    return prompt_pipeline_optimization


def save_chat_history(chat_history, curr_iter, output_folder):
    
    output_file = os.path.join(output_folder, f"chat_history_ver{curr_iter:03d}.txt")
    with open(output_file, "w") as file:
        for message in chat_history:
            file.write(f"{message['name']}: {message['content']}\n\n")

def save_seed_list(n, file_path, initial_seed):
    """Generates a list of deterministic integer seeds, saves them, and returns the list."""
    # Use Python's random module, seeded once, to generate other seeds
    seed_generator = random.Random(initial_seed)
    # Generate a list of n integers within a reasonable range
    int_seeds = [seed_generator.randint(0, 2**32 - 1) for _ in range(n)]

    with open(file_path, "w") as file:
        for seed_val in int_seeds:
            file.write(f"{seed_val}\n") # Save integer seeds

    print(f"Saved {n} integer seeds based on initial seed {initial_seed} to {file_path}")

    return int_seeds # Return the list of integers
    
def warm_start(function_definition_path: str, task_prompts: TaskPrompts, function_placeholder: str) -> str:
    '''
    Load the expert baseline function definition from a file and pass to the template executor.
    '''

    print("Performing warm start with expert baseline function")

    with open(function_definition_path, "r") as file:
        function_definition = file.read()
    
    executor = TemplatedLocalCommandLineCodeExecutor(
        template_script_func=task_prompts.run_pipeline_prompt,
        placeholder=function_placeholder,
        work_dir=work_dir,
        timeout=300
    )

    output = executor.execute_code_blocks([CodeBlock(code=function_definition, language="python")])

    print(output)
    

def save_run_info(args, run_output_dir, num_optim_iter, prompts_instance, cur_time, history_threshold, max_round, llm_model, n_top, n_worst, n_last):
     """Save comprehensive information about the run configuration."""
     # Create a dictionary with all the run information
     run_info = {
         "experiment_name": args.experiment_name,
         "dataset_path": args.dataset,
         "gpu_id": args.gpu_id,
         "random_seed": args.random_seed,
         "timestamp": cur_time,
         "num_optimization_iterations": num_optim_iter,
         "max_rounds": max_round,
         "history_threshold": history_threshold,
         "n_top": n_top,
         "n_worst": n_worst,
         "n_last": n_last,
         "llm_model": llm_model,
         "prompts_data": {
             "task_specific_prompts": {
                 "dataset_info": prompts_instance.dataset_info,
                 "task_details": prompts_instance.task_details,
                 "pipeline_metrics_info": prompts_instance.pipeline_metrics_info,
                 # "summary_prompt": prompts_instance.summary_prompt if hasattr(prompts_instance, 'summary_prompt') else None,
             },
             "agent_system_prompts": {
                 "code_writer": sys_prompt_code_writer(args.k, args.k_word),
                 # "code_verifier": sys_prompt_code_verifier,
             },
             "executable_pipeline_script_template": prompts_instance.run_pipeline_prompt(), # Call the method to get the script string
         }
     }

     # Write the run info to a JSON file
     with open(os.path.join(run_output_dir, "run_info.json"), "w") as file:
         json.dump(run_info, file, indent=4)

def update_run_info_with_end_timestamp(run_output_dir: str):
    """Adds the end timestamp to the run_info.json file."""
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_info_path = os.path.join(run_output_dir, "run_info.json")

    if os.path.exists(run_info_path):
        with open(run_info_path, "r") as file:
            run_info_data = json.load(file)
        
        run_info_data["timestamp_finish"] = end_time
        
        with open(run_info_path, "w") as file:
            json.dump(run_info_data, file, indent=4)
    else:
        print(f"Warning: {run_info_path} not found. Cannot add end timestamp.")



def create_latest_symlink(experiment_output_dir, run_output_dir):
     """Create a symlink to the latest run for easier access."""
     latest_link = os.path.join(experiment_output_dir, "latest")

     # Get just the directory name (timestamp) without the full path
     run_dir_name = os.path.basename(run_output_dir)

     # Remove existing symlink if it exists
     if os.path.islink(latest_link):
         os.unlink(latest_link)

     # Create relative symlink using just the directory name
     try:
         os.symlink(run_dir_name, latest_link)
         print(f"Created symlink: {latest_link} -> {run_dir_name}")
     except Exception as e:
         print(f"Failed to create symlink: {e}")

def main(args: argparse.Namespace):
    # Get current datetime once at the beginning
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create experiment-specific output directory
    experiment_output_dir = os.path.join(args.output, args.experiment_name)
    run_output_dir = os.path.join(experiment_output_dir, cur_time)
    # Create directories if they don't exist
    os.makedirs(run_output_dir, exist_ok=True)

    # Update output paths to use the new directory structure
    # output_function_bank = os.path.join(run_output_dir, "preprocessing_func_bank.json")
    output_function_bank = os.path.abspath(os.path.join(run_output_dir, "preprocessing_func_bank.json"))
    # Initialize function bank with empty list if it doesn't exist
    if not os.path.exists(output_function_bank):
        with open(output_function_bank, "w") as file:
            json.dump([], file)


    # Configuration
    my_gpu_id = args.gpu_id # GPU ID to use
    cache_seed = 4 # Cache seed for caching the results
    random_seed = args.random_seed # Random seed for reproducibility
    num_optim_iter = 30 # Number of optimization iterations
    max_round = 20  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10
    checkpoint_path = args.checkpoint_path
    # history_threshold = 5
    llm_model = "gpt-4.1" # Do not modify this string
    # llm_model = "gemini-2.5-pro"
    # llm_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    
    # Load task prompts
    if args.experiment_name == "spot_detection":
        from prompts.spot_detection_prompts import SpotDetectionPrompts, SpotDetectionPromptsWithSkeleton, _PREPROCESSING_FUNCTION_PLACEHOLDER
        prompt_class = SpotDetectionPromptsWithSkeleton
        sampling_function = lambda x: x['overall_metrics']['f1_score']
        kwargs_for_prompt_class = {"gpu_id": args.gpu_id, "seed": args.random_seed, "dataset_path": args.dataset, "function_bank_path": output_function_bank, "k": args.k, "k_word": args.k_word}
        # prompts = prompt_class(gpu_id=args.gpu_id, seed=args.random_seed, dataset_path=args.dataset, function_bank_path=output_function_bank)
        baseline_function_path = "prompts/spot_detection_expert.py.txt"
    elif args.experiment_name == "cellpose_segmentation":
        from prompts.cellpose_segmentation_prompts import CellposeSegmentationPrompts, CellposeSegmentationPromptsWithSkeleton, _PREPROCESSING_FUNCTION_PLACEHOLDER
        prompt_class = CellposeSegmentationPromptsWithSkeleton #CellposeSegmentationPrompts
        sampling_function = lambda x: x['overall_metrics']['average_precision']
        kwargs_for_prompt_class = {"gpu_id": args.gpu_id, "seed": args.random_seed, "dataset_path": args.dataset, "function_bank_path": output_function_bank}
        # prompts = prompt_class(gpu_id=args.gpu_id, seed=args.random_seed, dataset_path=args.dataset, function_bank_path=output_function_bank)
        baseline_function_path = "prompts/cellpose_segmentation_expert.py.txt"
    elif args.experiment_name == "medSAM_segmentation":
        from prompts.medsam_segmentation_prompts import MedSAMSegmentationPrompts, MedSAMSegmentationPromptsWithSkeleton, _PREPROCESSING_FUNCTION_PLACEHOLDER
        prompt_class = MedSAMSegmentationPromptsWithSkeleton #MedSAMSegmentationPrompts
        baseline_function_path = "prompts/medsam_segmentation_expert.py.txt"
        sampling_function = lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric']
        kwargs_for_prompt_class = {"gpu_id": args.gpu_id, "seed": args.random_seed, "dataset_path": args.dataset, "function_bank_path": output_function_bank, "checkpoint_path": checkpoint_path}

    else:
        raise ValueError(f"Experiment name {args.experiment_name} not supported")

    # Set GPU device
    # set_gpu_device(my_gpu_id)
    
    initial_prompts = prompt_class(**kwargs_for_prompt_class)
    save_run_info(args, run_output_dir, num_optim_iter, initial_prompts, cur_time, history_threshold=args.history_threshold, max_round=max_round, llm_model=llm_model, n_top=args.n_top, n_worst=args.n_worst, n_last=args.k)
    create_latest_symlink(experiment_output_dir, run_output_dir)
    
    seed_list_file = os.path.join(args.output,"seed_list.txt")
    # Generate seed list
    seed_list = save_seed_list(num_optim_iter, seed_list_file, args.random_seed)

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed, cache_path_root=f"{args.output}/cache") as cache:
        
        notes_shared = prepare_notes_shared(my_gpu_id, max_rounds=max_round)

        # Run baseline and insert to function bank first
        baseline_metric = ""
        if args.warm_start:
            warm_start(
                baseline_function_path,
                prompt_class(
                    gpu_id=args.gpu_id,
                    seed=0,
                    dataset_path=args.dataset,
                    function_bank_path=output_function_bank,
                ),
                _PREPROCESSING_FUNCTION_PLACEHOLDER
            )

            if args.metric_only:
                # Get baseline metric and reset function bank
                if args.experiment_name == "cellpose_segmentation":
                    baseline_metric = "Expert average precision score: "
                elif args.experiment_name == "medSAM_segmentation":
                    baseline_metric = "Expert DSC + NSD score: "
                elif args.experiment_name == "spot_detection":
                    baseline_metric = "Expert F1 score: "
                baseline_metric += str(sampling_function(last_n(output_function_bank, n=1)[0]))
                with open(output_function_bank, "w") as file:
                    json.dump([], file)

        for i in range(num_optim_iter):

            kwargs_for_prompt_class["seed"] = seed_list[i]
            prompts = prompt_class(**kwargs_for_prompt_class)
            
            executor_instance = TemplatedLocalCommandLineCodeExecutor(
                template_script_func=prompts.run_pipeline_prompt,
                placeholder=_PREPROCESSING_FUNCTION_PLACEHOLDER,
                work_dir=work_dir,
                timeout=300 * args.k
            )

            # Set up agents
            # code_executor_agent, code_writer_agent, code_verifier_agent, state_transition = set_up_agents(executor_instance)
            code_executor_agent, code_writer_agent, state_transition = set_up_agents(executor_instance, llm_model, args.k, args.k_word)

            

            group_chat = GroupChat(
                agents=[
                    code_executor_agent,
                    code_writer_agent,
                    # code_verifier_agent,
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


            prompt_pipeline_optimization = f"Agent Pipeline Seed {seed_list[i]} \n {prepare_prompt_pipeline_optimization(notes_shared, output_function_bank, prompts, sampling_function, i, history_threshold=args.history_threshold, total_iterations=num_optim_iter, n_top=args.n_top, n_worst=args.n_worst, n_last=args.k, baseline_metric=baseline_metric)}"
            
            chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_optimization, summary_method=None,
                                            # summary_args={"summary_prompt": prompts.summary_prompt},
                                            cache=cache)
            save_chat_history(chat_result.chat_history, i, run_output_dir)

    update_run_info_with_end_timestamp(run_output_dir)
    if args.experiment_name == "cellpose_segmentation":
        os.system(f"python figs/cellpose_analyze_trajectories.py --json_path {output_function_bank} --data_path {args.dataset} --device {args.gpu_id}")
    elif args.experiment_name == "medSAM_segmentation":
        # os.system(f"python figs/medsam_analyze_trajectories.py --json_path {output_function_bank} --data_path {args.dataset} --device {args.gpu_id}")
        pass
    elif args.experiment_name == "spot_detection":
        # os.system(f"python figs/spot_detection_analyze_trajectories.py --json_path {output_function_bank} --data_path {args.dataset} --device {args.gpu_id}")
        pass

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
        required=False,
        help="The working directory for the agent to access source code."
    )

    parser.add_argument(
        "--warm_start",
        action='store_true'
    )

    parser.add_argument(
        "--metric_only",
        action="store_true"
    )
    
    parser.add_argument(
        "--n_top",
        type=int,
        default=0,
        help="Number of top functions to show in the function bank."
    )

    parser.add_argument(
        "--n_worst",
        type=int,
        default=0,
        help="Number of worst functions to show in the function bank."
    )

    # parser.add_argument(
    #     "--n_last",
    #     type=int,
    #     default=5,
    #     help="Number of last functions to show in the function bank."
    # )

    parser.add_argument(
        "--history_threshold",
        type=int,
        default=0,
        help="Number of history threshold to show in the function bank."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        required=False,
        help="Preprocessing function group size."
    )

    parser.add_argument(
        "--k_word",
        type=str,
        default="five",
        required=False,
        help="Preprocessing function group size in English."
    )

    args = parser.parse_args()

    # Work directory
    if args.work_dir is None:
        work_dir = args.output
    else:
        work_dir = args.work_dir
    # server = LocalJupyterServer(log_file=os.path.join("..", args.output, "jupyter_gateway.log"))
    # server = LocalJupyterServer(log_file=None)
    # executor = JupyterCodeExecutor(server, output_dir=args.output, timeout=300) # very high timeout for long running tasks
    # executor = LocalCommandLineCodeExecutor(work_dir=work_dir, timeout=300)


    # if args.experiment_name == "spot_detection":
    #     from prompts.spot_detection_prompts import SpotDetectionPrompts
    #     prompt_class = SpotDetectionPrompts
    # elif args.experiment_name == "cellpose_segmentation":
    #     from prompts.cellpose_segmentation_prompts import CellposeSegmentationPrompts, CellposeSegmentationPromptsWithSkeleton, _PREPROCESSING_FUNCTION_PLACEHOLDER
    #     prompt_class = CellposeSegmentationPromptsWithSkeleton #CellposeSegmentationPrompts
    # elif args.experiment_name == "medSAM_segmentation":
    #     from prompts.medsam_segmentation_prompts import MedSAMSegmentationPrompts, MedSAMSegmentationPromptsWithSkeleton, _PREPROCESSING_FUNCTION_PLACEHOLDER
    #     prompt_class = MedSAMSegmentationPromptsWithSkeleton
    # else:
    #     raise ValueError(f"Experiment name {args.experiment_name} not supported")

    # executor = TemplatedLocalCommandLineCodeExecutor(template_script_func=prompt_class, placeholder=_PREPROCESSING_FUNCTION_PLACEHOLDER, work_dir=work_dir, timeout=300)


    main(args)