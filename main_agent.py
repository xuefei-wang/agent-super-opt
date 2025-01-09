from dotenv import load_dotenv

load_dotenv()

import os
import torch

from autogen import OpenAIWrapper
from autogen.coding import CodeBlock
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from autogen import ConversableAgent, GroupChat, GroupChatManager

from src.prompts import (
    sys_prompt_code_writer,
    sys_prompt_code_verifier,
    sys_prompt_visual_critic,
    prompt_denoise,
    prompt_find_whole_cell_channels,
)


def set_gpu_device(gpu_id: int) -> None:
    """Set global GPU device for both PyTorch and TensorFlow."""
    # Set CUDA visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Tensorflow (used by Mesmer) specific
    os.environ["TF_CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set PyTorch default device
    torch.cuda.set_device(gpu_id)


# Set the GPU device to use
set_gpu_device(3)

max_round = 100  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10

server = LocalJupyterServer()
executor = JupyterCodeExecutor(server, output_dir="output")

code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={
        "executor": executor
    },  # Use the docker command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
    # is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
)


code_writer_agent = ConversableAgent(
    "code_writer",
    system_message=sys_prompt_code_writer,
    llm_config={
        "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
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


# builder_model = OpenAIWrapper(
#     config_list=[{
#         "model": "gpt-4o-mini",
#         "api_key": os.environ["OPENAI_API_KEY"],
#         "api_type": "openai",
#     }]
# )

# sys_prompt_visual_critic = builder_model.create(
#     messages=[
#         {
#             "role": "user",
#             "content": f"""
#             Generate a system message for a visual inspector who is evaluating an image analysis task.
#             The images are already provided to the visual inspector.
#             You goal is to guide the visual inspector on how to evaluate the results using the following template:

#             # Task Description
#             ...

#             # Evaluation aspects
#             ...

#             """,
#         }
#     ]
# ).choices[0].message.content


visual_critic_agent = MultimodalConversableAgent(
    "visual_critic",
    system_message=sys_prompt_visual_critic,
    llm_config={
        "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
    },
)

# def message_hook(messages):
#     """Hook that processes messages"""
#     # Check if the last message contains an image
#     has_image = False
#     for m in messages:
#         if m['type'] == 'image_url':
#             has_image = True
#             break

#     if has_image:
#         messages.append(
#             {
#                 "type": "text",
#                 "text": f"This is a image and it is a visualization from {sometask}. Describe what you see. Does the result look correct?",
#             }
#         )

#     return messages


# visual_critic_agent.register_hook("process_last_received_message", message_hook)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if len(messages) <= 1:  # First round
        return code_writer_agent

    if last_speaker is code_writer_agent:
        if "QUERY_INSPECTOR:" in messages[-1]["content"]:
            return visual_critic_agent
        return code_verifier_agent
    elif last_speaker is code_verifier_agent:
        return code_executor_agent
    elif last_speaker is code_executor_agent:
        if "exitcode: 1" in messages[-1]["content"]:
            return code_writer_agent
        # elif "SAVED_AT=<img" in messages[-1]["content"]:
        #     return visual_critic_agent
        else:
            return code_writer_agent
    elif last_speaker is visual_critic_agent:
        if "TERMINATE" in messages[-1]["content"]:
            return None
        else:
            return code_writer_agent


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


#######################################################################################################################################################

with open("artifacts/docs.md", "r") as file:
    documentation = file.read()


dataset_info = """
```markdown
This is a single-channel nuclear segmentation dataset. It consists of images from different experiments, different settings - a heterogenous dataset covering
five different cell lines (NIH-3T3, HeLa-S3, HEK293, RAW 264.7, and PC-3).
```
"""

# dataset_path = "/data/user-data/xwang3/DynamicNuclearNet/DynamicNuclearNet-segmentation-v1_0/val.npz"
dataset_path = "/home/julie/Downloads/DynamicNuclearNet-segmentation-v1_0/val.npz"

notes = """
- For SAM-2, please run the following code snippet to set up the model config and checkpoint path:
    ```python
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    # Initialize Hydra config path to bypass SAM-2's settings
    if not GlobalHydra().is_initialized():
        hydra.initialize(config_path="src/sam2/sam2/configs", version_base="1.2")

    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent
    model_cfg = "sam2.1/sam2.1_hiera_t.yaml"  # Config path has already been set to it parent folder
    checkpoint_path = str(PROJECT_ROOT / "src/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    ```

"""


prompt_pipeline = f"""
# Cell Segmentation Analysis Pipeline
Develop an end-to-end pipeline for cell segmentation analysis.

## About the dataset: 

{dataset_info}

And it's located at: {dataset_path}

## Task Details:

Here is the description of the pipeline steps. You will work with a visual critic to optimize the pipeline iteratively.

1. Load the data using the provided APIs. 

2. Select 3 diverse visual samples and visualize them using the provided APIs. You will be working with these 3 samples for the rest of the pipeline.

3. Implement a custom denoising module to clean the image data. You'll write this module from scratch following these specifications and apply it to the 3 samples:
    {prompt_denoise}. 

4. Use SAM-2 segmenter to generate masks, calculate metrics, and create visualizations (both raw images and gt-predited masks comparison).

5. Put together and clean up the code snippets and save each version of your code to `output/pipeline_N.py`, where N is the iteration number (starting from 1).

6. Give the results to a visual cirtic for feedback. You should also provide a brief description the pipeline. In summary, the results you provide should include:
    - path to raw image visualization
    - path to groundtruth-prediction mask comparison
    - average metrics
    - brief description of the functions you used and their adjustable parameters

7. Once received the feedback from the visual critic, you should update the pipeline accordingly and redo the analysis. Repeate this procedure 
at least 5 times. Each time, save your code to `output/pipeline_N.py` file, keep track of the number of iterations.

## Available APIs:

```markdown 
{documentation}
```

## Additional Notes:

{notes}

"""


#######################################################################################################################################################

code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline)
