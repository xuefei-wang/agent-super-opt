
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class TaskPrompts:
    """Task prompts for a task."""

    gpu_id : int
    seed : int
    function_bank_path : str
    dataset_path : str

    dataset_info : str
    summary_prompt : str
    task_details : str
    pipeline_metrics_info : str
    @abstractmethod
    def run_pipeline_prompt(self) -> str:
        pass
    
    @abstractmethod
    def save_function_prompt(self) -> str:
        pass
