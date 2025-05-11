
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
    k : int
    k_word : str
    advantage_enabled : bool = False
    checkpoint_path : Optional[str] = None
    dataset_size :Optional[int] = None
    batch_size : Optional[int] = None
    
    @abstractmethod
    def run_pipeline_prompt(self) -> str:
        pass
    
    @abstractmethod
    def save_function_prompt(self) -> str:
        pass

    @abstractmethod
    def get_task_details(self):
        pass

    @abstractmethod
    def get_pipeline_metrics_info(self):
        pass

    def if_advantage(self, enabled_string, disabled_string=""):
        return enabled_string if self.advantage_enabled else disabled_string