import random

class ProbabilisticExploration:
    '''
    Class for wrapping different modes of agent exploration

    Attributes:
        seed (int): Random seed for sampling
        temperature (float): Temperature for sampling: higher values lead to more exploration
    '''

    explore_prompt = """
    Prioritize exploring new functions, taking inspiration from the best or the most recent functions in the function bank. Do NOT to use the same methods as functions already in the pre-processing function bank.
    For example, if the best or worst functions use filtering, use a different operation altogether. If you do use the same methods, try to use them in a different way. 
    """

    refine_prompt = """
    Prioritize refining existing functions, choosing either the best or most recent functions and suggesting an incremental improvement, i.e. adjusting hyperparameters.
    For example, if a function uses an operation with a kernel size of 5, suggest a kernel size of 3 or 7.
    """

    def __init__(self, seed: int, temperature: float):
        self.seed = seed
        self.temperature = temperature
        random.seed(seed)

    def __post_init__(self):
        '''Verify that the temperature is between 0 and 1'''
        if not (0 <= self.temperature <= 1):
            raise ValueError(f"Temperature must be between 0 and 1, got {self.temperature}")
    
    def get_prompt(self) -> str:
        '''Return the randomly sampled prompt based on the temperature'''
        if random.random() < self.temperature:
            return self.explore_prompt
        else:
            return self.refine_prompt