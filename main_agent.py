from dotenv import load_dotenv
import os
import torch

from autogen import OpenAIWrapper, Cache, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import CodeBlock
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)

from src.prompts import (
    sys_prompt_code_writer,
    sys_prompt_code_verifier,
    sys_prompt_visual_critic,
    prompt_denoise,
    prompt_find_whole_cell_channels,
)

# Load environment variables
load_dotenv()

def set_gpu_device(gpu_id: int) -> None:
    """Set global GPU device for both PyTorch and TensorFlow."""
    # Set CUDA visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Tensorflow (used by Mesmer) specific
    os.environ["TF_CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set PyTorch default device
    torch.cuda.set_device(gpu_id)


def set_up_agents(max_round):
    server = LocalJupyterServer()
    executor = JupyterCodeExecutor(server, output_dir="output", timeout=600)

    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,
        code_execution_config={
            "executor": executor
        },  # Use the docker command line code executor
        human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
        # is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    )

    code_writer_agent = ConversableAgent(
        "code_writer",
        system_message=sys_prompt_code_writer,
        llm_config={
            "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
        },
        code_execution_config=False,
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
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    visual_critic_agent = MultimodalConversableAgent(
        "visual_critic",
        system_message=sys_prompt_visual_critic,
        llm_config={
            # "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
            "config_list": [{"model": "gemini-1.5-flash", "api_key": os.environ["GEMINI_API_KEY"], "api_type": "google"}],
            # "config_list": [{"model": "gemini-2.0-flash-exp", "api_key": os.environ["GEMINI_API_KEY"], "api_type": "google"}],
            "cache_seed": None,
        },
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

        if last_speaker is code_writer_agent:
            if "QUERY_CRITIC_REPORT:" in messages[-1]["content"]:
                return visual_critic_agent
            return code_verifier_agent
        elif last_speaker is code_verifier_agent:
            return code_executor_agent
        elif last_speaker is code_executor_agent:
            return code_writer_agent
        elif last_speaker is visual_critic_agent:
            if "TERMINATE" in messages[-1]["content"]:
                return None
            return code_writer_agent

    # Set up group chat
    group_chat = GroupChat(
        agents=[
            code_executor_agent,
            code_writer_agent,
            code_verifier_agent,
            visual_critic_agent,
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
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        is_termination_msg=lambda msg: (
            "TERMINATE" in msg["content"] if msg["content"] else False
        ),
    )

    return code_executor_agent, group_chat_manager

# Load documentation and dataset information
with open("artifacts/docs.md", "r") as file:
    documentation = file.read()

dataset_info = """
```markdown
This is a single-channel nuclear segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset covering
five different cell lines (NIH-3T3, HeLa-S3, HEK293, RAW 264.7, and PC-3).
```
"""

dataset_path = "/data/user-data/xwang3/DynamicNuclearNet/DynamicNuclearNet-segmentation-v1_0/val.npz"
# dataset_path = "/home/julie/Downloads/DynamicNuclearNet-segmentation-v1_0/val.npz"

def prepare_notes_shared(my_gpu_id):
    notes_shared = f"""
    - Always check the documentation for the available APIs before reinventing the wheel
    - Use GPU {my_gpu_id} for running the pipeline
    """
    return notes_shared


notes_pipeline_development = f"""
- For SAM-2, please run the following code snippet to set up the model config and checkpoint path. Let's run this code snippet and instantiate the model before starting the pipeline.
    ```python
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="src/sam2/sam2/configs", version_base="1.2")

    from pathlib import Path

    PROJECT_ROOT = Path(os.getcwd())
    model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
    checkpoint_path = str(PROJECT_ROOT / "src/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    ```
    Don't modify any of these code! 

- Always check the documentation for the available APIs before reinventing the wheel
- The images should be saved to the `output` directory
"""


notes_pipeline_optimization = f"""
- The image paths will be returned to you after the code execution so that you can collect them and send them to the visual critic for feedback
"""



def prepare_prompt_pipeline_development(notes_shared, notes_pipeline_development):
    prompt_pipeline_development = f"""
    # Cell Segmentation Analysis Pipeline Development
    ## Objective:
    Develop an end-to-end pipeline for cell segmentation analysis.

    ## About the dataset: 
    {dataset_info}
    Location: {dataset_path}

    ## Task Details:
    Here is the description of the pipeline steps. You will work with a visual critic to optimize the pipeline iteratively.

    1. Load the data using the provided APIs. 
    2. Select 2 samples (the first and the last) as you will be working with a small subset of the data for testing purposes.
    3. Use SAM-2 segmenter to generate masks, calculate metrics, and create visualizations (both raw images and gt-predited mask comparisons).
    4. Reflect on the conversation history, collect the code snippets that were actually useful for developing the pipeline and summarize them into a pure code format.

    ## Available APIs:
    ```markdown 
    {documentation}
    ```

    ## Additional Notes:
    {notes_shared}
    {notes_pipeline_development}
    """

    return prompt_pipeline_development


def prepare_prompt_pipeline_optimization(last_summary, notes_shared):

    prompt_pipeline_optimization = f"""
    # Cell Segmentation Analysis Pipeline Optimization
    ## Objective:
    Optimize the existing pipeline for cell segmentation analysis by incorporating feedback from the visual critic.

    ## About the dataset: 
    {dataset_info}
    Location: {dataset_path}

    ## Task Details:
    You will be provided with a script of the pipeline that performs cell segmentation analysis. Your task is to optimize the pipeline by incorporating feedback from the visual critic.

    1. Execute the pipeline script, collect the results from execution output, and format into a report with the title 'QUERY_CRITIC_REPORT'. It will be automatically sent to the visual critic for feedback.
    2. Once received the feedback from the visual critic, implement the changes and update the pipeline accordingly.

    ## Pipeline Developed So Far:
    {last_summary}

    ## Available APIs:
    ```markdown
    {documentation}
    ```

    ## Additional Notes:
    {notes_shared}
    {notes_pipeline_optimization}
    """

    return prompt_pipeline_optimization


def save_pipeline_script(pipeline_script, curr_iter):
    with open(f"output/pipeline_script_V{curr_iter:03d}.py", "w") as file:
        file.write(pipeline_script)


def main():
    # Configuration
    my_gpu_id = 7 # GPU ID to use
    cache_seed = 1234  # Cache seed for caching the results
    num_optim_iter = 3 # Number of optimization iterations
    max_round = 100  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10

    # Set GPU device
    set_gpu_device(my_gpu_id)

    # Set up agents
    code_executor_agent, group_chat_manager = set_up_agents(max_round=max_round)

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed) as cache:
        # Pipeline development
        notes_shared = prepare_notes_shared(my_gpu_id)
        prompt_pipeline_development = prepare_prompt_pipeline_development(notes_shared, notes_pipeline_development)
        chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_development, summary_method="reflection_with_llm",
                                        summary_args={"summary_prompt": "Summarize the pipeline you developed into pure code format."})
        last_summary = chat_result.summary
        save_pipeline_script(last_summary, 0)

        # Pipeline optimization
        for i in range(1, num_optim_iter):
            prompt_pipeline_optimization = prepare_prompt_pipeline_optimization(last_summary, notes_shared)
            
            chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_optimization, summary_method="reflection_with_llm",
                                            summary_args={"summary_prompt": "Summarize the pipeline you optimized into pure code format."},)
            last_summary = chat_result.summary
            save_pipeline_script(last_summary, i)

if __name__ == "__main__":
    main()