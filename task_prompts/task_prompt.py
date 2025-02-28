
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class TaskPrompts:
    """Task prompts for a task."""

    gpu_id : int
    seed : int

    dataset_info : str
    dataset_path : str
    summary_prompt : str
    save_to_function_bank_prompt : str
    task_details : str

    @abstractmethod
    def run_pipeline_prompt(self) -> str:
        pass

