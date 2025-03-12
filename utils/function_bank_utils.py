

def top_n(function_bank: list, sorting_key: str, n: int = 10):
    sorted_bank = sorted(function_bank, key=lambda x: x["sorting_key"])
    
    return sorted_bank[:n]