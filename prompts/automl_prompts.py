_AUTOML_PARAMETERIZED_FUNCTION_PLACEHOLDER = "# --- AUTOML_PARAMETERIZED_FUNCTIONS_INSERT ---"

def sys_prompt_automl_agent(n_functions: int):
    """
    System prompt for AutoML agent.

    Args:
        n_functions (int): Number of function pairs to optimize
    """
    return f"""
You are an AutoML optimization specialist focused on converting image preprocessing and postprocessing functions into Optuna-optimized versions.

Your role is to take existing high-performing functions and make their numeric parameters tunable through hyperparameter optimization using Optuna's trial.suggest_* API.

**Core Responsibilities:**
1. Analyze function code to identify optimizable numeric parameters (thresholds, kernel sizes, iterations, etc.)
2. Replace hardcoded values with appropriate Optuna trial.suggest_* calls
3. Choose reasonable parameter ranges based on the operation type
4. Ensure all parameter names are unique across all functions using function index prefixes (e.g., `f1_kernel_size`, `f2_threshold`)
5. Preserve the original algorithmic structure and function signatures

**Optuna API Reference:**
- `trial.suggest_int(name, low, high)` - for integer parameters
- `trial.suggest_float(name, low, high)` - for float parameters
- `trial.suggest_categorical(name, choices)` - for categorical/boolean parameters

**Critical Requirements:**
- The `trial` object is available in global scope - do NOT add it as a function parameter
- Output exactly {n_functions * 2} individual function definitions in a single markdown code block (```python ... ```)
- Functions must be enumerated: `preprocess_images_1`, `preprocess_images_2`, ..., `preprocess_images_{n_functions}` and `postprocess_preds_1`, `postprocess_preds_2`, ..., `postprocess_preds_{n_functions}`
- Each function pair should have unique parameter names with index prefix
- Only output function definitions - no data loading, evaluation, or other code
- Code must be in a markdown code block to be executed

**Workflow:**
1. Receive feedback from code execution
2. If errors occur, fix the functions and output all {n_functions} pairs in a single code block
3. Once all {n_functions} function pairs are successfully evaluated, print metrics in format: `preprocess_images_<i> & postprocess_preds_<i>: <metric>: <score>`
4. After successful evaluation of all pairs, write "TERMINATE"
"""


def prepare_automl_prompt(function_bank_path: str, n_functions: int = 3, sorting_function = None):
    """
    Prepare the prompt for the AutoML agent with function bank context.

    Args:
        function_bank_path (str): Path to the function bank JSON file
        n_functions (int): Number of top functions to include for optimization
        sorting_function (callable): Function to sort the function bank entries

    Returns:
        str: Complete prompt for the AutoML agent
    """
    from utils.function_bank_utils import top_n, pretty_print_list
    import os

    # Get top performing functions from function bank
    function_bank_sample = ""
    if os.path.exists(function_bank_path):
        try:
            if sorting_function is None:
                raise ValueError("sorting_function must be provided")

            top_functions = top_n(function_bank_path, n=n_functions,
                                sorting_function=sorting_function)

            if top_functions:
                function_bank_sample = f"""
## Top {n_functions} performing functions from function bank:
{pretty_print_list(top_functions)}
"""
            else:
                function_bank_sample = "Function bank is empty or no functions available for optimization."
        except Exception as e:
            function_bank_sample = f"Error reading function bank: {e}"
    else:
        function_bank_sample = "Function bank file not found."

    prompt = f"""
Your task is to create {n_functions} Optuna-optimized function pairs from the best-performing preprocessing and postprocessing functions in the function bank.

{function_bank_sample}

## Instructions:
1. Above are the top {n_functions} **entries** from the function bank
2. Each entry contains one preprocessing function (`preprocess_images`) and one postprocessing function (`postprocess_preds`)
3. Note: the functions themselves are NOT enumerated, but the entries are numbered (Entry 1, Entry 2, etc.)
4. You must create {n_functions} enumerated function pairs based on these entries:
   - Entry 1 → create `preprocess_images_1` and `postprocess_preds_1`
   - Entry 2 → create `preprocess_images_2` and `postprocess_preds_2`
   - Entry {n_functions} → create `preprocess_images_{n_functions}` and `postprocess_preds_{n_functions}`
5. For each function, identify numeric parameters that can be optimized (constants, thresholds, kernel sizes, etc.)
6. Replace hardcoded numeric values with Optuna trial.suggest_* calls
7. Ensure each parameter has a unique name with function index prefix (e.g., `f1_kernel_size`, `f2_threshold`)
8. Use appropriate parameter ranges and distributions
9. Maintain the exact same function signatures and algorithmic behavior

## CRITICAL: Output Format Requirements:
- You MUST output exactly {n_functions * 2} individual function definitions in a single code block
- Preprocessing functions: `preprocess_images_1`, `preprocess_images_2`, ..., `preprocess_images_{n_functions}`
- Postprocessing functions: `postprocess_preds_1`, `postprocess_preds_2`, ..., `postprocess_preds_{n_functions}`
- Do NOT output tuples, pairs, or any other data structures - only individual function definitions

## Parameter Guidelines:
- **Kernel sizes**: Usually odd integers, range 3-15
- **Thresholds**: Float values, typically 0.0-1.0 or image-specific ranges
- **Iterations**: Integer values, typically 1-10
- **Scaling factors**: Float values, typically 0.5-2.0
- **Blur parameters**: Float values for sigma, int values for kernel size
- **Parameter names must include function index**: e.g., `f1_kernel_size`, `f2_threshold`, etc.

## Expected Output:
Generate exactly {n_functions} complete function pairs (preprocessing + postprocessing) that:
1. Are properly enumerated with indices (_1, _2, ..., _{n_functions})
2. Incorporate Optuna optimization with trial.suggest_* calls
3. Maintain the performance characteristics of the original functions
4. Have unique parameter names across all function pairs
"""

    return prompt