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


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if len(messages) <= 1:  # First round
        return code_writer_agent

    if last_speaker is code_writer_agent:
        return code_verifier_agent
    elif last_speaker is code_verifier_agent:
        return code_executor_agent
    elif last_speaker is code_executor_agent:
        return code_writer_agent
    else:
        return code_writer_agent


group_chat = GroupChat(
    agents=[
        code_executor_agent,
        code_writer_agent,
        code_verifier_agent,
        # visual_inspector_agent,
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
# Title
A spatially resolved timeline of the human maternal–fetal interface
## Abstract
Beginning in the first trimester, fetally derived extravillous trophoblasts (EVTs) invade the uterus and remodel its spiral arteries, 
transforming them into large, dilated blood vessels. Several mechanisms have been proposed to explain how EVTs coordinate with the maternal 
decidua to promote a tissue microenvironment conducive to spiral artery remodelling (SAR). However, it remains a matter of debate regarding 
which immune and stromal cells participate in these interactions and how this evolves with respect to gestational age. Here we used a 
multiomics approach, combining the strengths of spatial proteomics and transcriptomics, to construct a spatiotemporal atlas of the human
maternal–fetal interface in the first half of pregnancy. We used multiplexed ion beam imaging by time-of-flight and a 37-plex antibody panel
to analyse around 500,000 cells and 588 arteries within intact decidua from 66 individuals between 6 and 20 weeks of gestation, integrating
this dataset with co-registered transcriptomics profiles. Gestational age substantially influenced the frequency of maternal immune and stromal 
cells, with tolerogenic subsets expressing CD206, CD163, TIM-3, galectin-9 and IDO-1 becoming increasingly enriched and colocalized at later 
time points. By contrast, SAR progression preferentially correlated with EVT invasion and was transcriptionally defined by 78 gene ontology 
pathways exhibiting distinct monotonic and biphasic trends. Last, we developed an integrated model of SAR whereby invasion is accompanied by 
the upregulation of pro-angiogenic, immunoregulatory EVT programmes that promote interactions with the vascular endothelium while avoiding the 
activation of maternal immune cells.
```
"""

dataset_path = "/data/user-data/xwang3/celltype-data-public-greenbaum-with-nuclei-channel/Tissue-Reproductive-Greenbaum_Uterus_MIBI.zarr"

notes = """
- Just load and process the the following two images from the dataset:
    - "HBM555.ZNTK.962-1d988a3a09dd6ab19d70f6c33974eef2"
    - "HBM786.VLVN.435-2d6d2131e94f60aff982c3d52a9fcb27"

- No need to run visualization for now

- Plot progress bar whenever possible

- the model weights of PhenotyperDeepCellTypes can be found at "pretrained/model_c_patch2_skip_Greenbaum_Uterus_0.pt"
"""


prompt_pipeline = f"""
# Cell Phenotyping Analysis Pipeline
Your task is to develop and run a complete cell-phenotyping analysis pipeline on a maternal-fetal interface dataset. The dataset is located at {dataset_path} and contains multiple images.

## About the dataset: 

{dataset_info}

## Task Details:

The pipeline consists of several steps, including denoising, channel selection, segmentation, and phenotyping.

First, you'll need to load the imaging data using the provided APIs.

Next, you'll need to implement a custom denoising module to clean the image data. You'll write this module from scratch following these specifications:
    {prompt_denoise}

After denoising, you'll need to carefully select the optimal imaging channels for whole-cell segmentation. This step requires reasoning about which markers will best identify both nuclear and membrane features following the instructions below:
    {prompt_find_whole_cell_channels}

With images denoised and optimal channels selected, you'll then:
- Run the segmentation model to identify individual cells
- Apply the DeepCell Types model for cell phenotyping
Both of these models are provided through APIs.

## Available APIs :

see documentation:
```markdown 
{documentation}
```

## Additional Notes:

{notes}

"""


#######################################################################################################################################################

code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline)
