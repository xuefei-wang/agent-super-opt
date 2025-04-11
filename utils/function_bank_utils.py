import json
from typing import Callable

def pretty_print_list(lst: list) -> str:
    '''Pretty print a subset of the function bank'''
    result = ""
    for item in lst:
        for key, value in item.items():
            if key == 'preprocessing_function':
                continue
            else:
                result += f"{key}: {value}\n"
        
        result += f"```python\n{item['preprocessing_function']}\n```\n"
    
    return result

def top_n(function_bank_path: str, sorting_function: Callable[[dict], float], n: int = 5) -> list:
    '''Return top N functions from function bank as a list'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        # Filter out None values
        sorted_bank = list(filter(lambda x:sorting_function(x) is not None, json_array))
        sorted_bank = sorted(sorted_bank, key=lambda x: sorting_function(x))
        return sorted_bank[:n]
    
def worst_n(function_bank_path: str, sorting_function: Callable[[dict], float], n: int = 5) -> list:
    '''Return worst N functions from function bank as a list'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        # Filter out None values
        sorted_bank = list(filter(lambda x:sorting_function(x) is not None, json_array))
        sorted_bank = sorted(json_array, key=lambda x:sorting_function(x), reverse=True)
        return sorted_bank[:n]

def last_n(function_bank_path: str, n: int = 5) -> list:
    '''Return last N functions from function bank as a list'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        return json_array[-n:]