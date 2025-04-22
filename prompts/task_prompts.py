
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
    grpo_k : int

    dataset_info : str
    summary_prompt : str
    pipeline_prompt : str
    task_details : str
    pipeline_metrics_info : str
    grpo_k_word : str
    checkpoint_path : Optional[str] = None
    save_to_function_bank_prompt : str = """
        ```python
        import inspect
        import json
        
        
        def write_results(preprocessing_fns, metrics_list):
            '''
            Write the results of evaluation to the function bank JSON.
        
            Requires:
            preprocessing_fns: list of {k_word} preprocessing functions
            metrics_list: list of the metrics for each preprocessing function
            '''
        
            with open('{function_bank_path}', 'r') as file:
                json_array = json.load(file)
        
            with open('{function_bank_path}', 'w') as file:
                json_data = [{{**fn_metrics, "preprocessing_function": inspect.getsource(preprocessing_fn)}} for fn_metrics, preprocessing_fn in zip(metrics_list, preprocessing_fns)]
                json_array.extend(json_data)
                json.dump(json_array, file)
        
            print("Finished writing preprocessing functions to function bank")
        ```
    """
    pre_process_func_stub : str = """
            def {signature}(images: ImageData) -> ImageData:
                # YOUR CODE HERE
                pass"""

    @abstractmethod
    def run_pipeline_prompt(self) -> str:
        function_signatures = []
        function_stubs = []
        for i in range(1, self.k + 1):
            function_signature_i = f"preprocess_images_{i}"
            function_stub_i = self.pre_process_func_stub.format(signature=function_signature_i)
            function_signatures.append(function_signature_i)
            function_stubs.append(function_stub_i)
        joined_names = ", ".join(function_signatures)
        joined_stubs = "\n".join(function_stubs)
        prompt = self.pipeline_prompt
        prompt = prompt.replace("{gpu_id}", str(self.gpu_id))
        prompt = prompt.replace("{seed}", str(self.seed))
        prompt = prompt.replace("{data_path}", self.dataset_path)
        prompt = prompt.replace("{k_word}", self.grpo_k_word)
        prompt = prompt.replace("{functions}", joined_stubs)
        prompt = prompt.replace("{function_names}", joined_names)
        return prompt

    @abstractmethod
    def save_function_prompt(self) -> str:
        prompt = self.save_to_function_bank_prompt
        prompt = prompt.replace("{function_bank_path}", self.function_bank_path)
        prompt = prompt.replace("{k_word}", self.grpo_k_word)
        return prompt

    @abstractmethod
    def task_details_prompt(self) -> str:
        prompt = self.task_details
        prompt = prompt.replace("{k_word}", self.grpo_k_word)
        return prompt