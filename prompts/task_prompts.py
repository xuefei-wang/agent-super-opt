
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskPrompts:
    """Task prompts for a task."""

    gpu_id : int
    seed : int
    function_bank_path : str
    dataset_path : str

    dataset_info : str
    # summary_prompt : str
    task_details : str
    pipeline_metrics_info : str
    k : int
    k_word : str
    checkpoint_path : Optional[str] = None

    @abstractmethod
    def run_pipeline_prompt(self) -> str:
        pass
    
    @abstractmethod
    def save_function_prompt(self) -> str:
        pass

    def get_task_details(self) -> str:
        return self.task_details.format(k=self.k, k_word=self.k_word)