import ast
import types
import inspect
import optuna
import json

import cv2 as cv
import numpy as np
from src.data_io import ImageData
# from src.spot_detection import DeepcellSpotsDetector
# from src.cellpose_segmentation import CellposeTool
# from src.medsam_segmentation import MedSAMTool
from assets.opencv_arg_rules import OPENCV_ARG_RULES
from prompts.task_prompts import TaskPrompts
import logging
import traceback
import os

def int_to_param_name(n):
    if n < 10:
        return f"param_{n}" 
    else:
        n -= 10
        return f"param_{chr(97 + n)}"


class CV2ParamExtractor(ast.NodeTransformer):
    def __init__(self):
        self.param_counter = 0
        self.param_info = []
        self.original_values = {}

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"cv", "cv2"}
        ):
            func_name = node.func.attr
            # If we don't have rules for this function, skip it
            if func_name not in OPENCV_ARG_RULES:
                return self.generic_visit(node)
            
            rules = OPENCV_ARG_RULES.get(func_name, {}).get("args", {})

            new_args = []
            for idx, arg in enumerate(node.args):
                rule = rules.get(idx)
                new_args.append(self.replace_constants(arg, rule))
            node.args = new_args
        return self.generic_visit(node)

    def replace_constants(self, node, rule=None):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            ptype = "float" if isinstance(node.value, float) else "int"
            param_name = int_to_param_name(self.param_counter)
            constraints = rule.get("constraints", {}) if rule else {}
            self.param_info.append({
                "name": param_name,
                "type": ptype,
                "constraints": constraints
            })
            self.original_values[param_name] = node.value
            self.param_counter += 1
            return ast.Name(id=param_name, ctx=ast.Load())
        elif isinstance(node, ast.Tuple):
            elements = []
            values = []
            for i, el in enumerate(node.elts):
                el_rule = None
                if rule and rule["type"].startswith("tuple") and "int" in rule["type"]:
                    el_rule = {"type": "int", "constraints": rule.get("constraints", {})}
                new_el = self.replace_constants(el, el_rule)
                values.append(el.value if isinstance(el, ast.Constant) else None)
                elements.append(new_el)

            param_name = int_to_param_name(self.param_counter)
            self.param_info.append({
                "name": param_name,
                "type": rule["type"] if rule else "tuple[int,int]",
                "constraints": rule.get("constraints", {}) if rule else {}
            })
            self.original_values[param_name] = tuple(values)
            self.param_counter += 1
            return ast.Name(id=param_name, ctx=ast.Load())
        return node

def transform_opencv_constants(func_str):
    """
    Transform OpenCV function calls in a string to use parameters instead of constants.
    Args:
        func_str (str): The function string to transform.
    Returns:
        tuple: A tuple containing the modified function string, parameter information, and original values.
    """
    tree = ast.parse(func_str)
    transformer = CV2ParamExtractor()
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree), transformer.param_info, transformer.original_values


def define_optuna_search_space(trial, param_info):
    params = {}
    for param in param_info:
        name = param['name']
        param_type = param['type']
        constraints = param['constraints']
        
        if 'enum' in constraints:
            params[name] = trial.suggest_categorical(name, constraints['enum']) 
        elif param_type == 'int':
            low = constraints.get('min', 1)
            high = constraints.get('max', 11)
            if constraints.get('odd'):
                # Only odd integers
                candidates = list(range(low | 1, high + 1, 2))  # bitwise OR for odd start
                params[name] = trial.suggest_categorical(name, candidates)
            else:
                params[name] = trial.suggest_int(name, low, high)
        elif param_type == 'float':
            low = constraints.get('min', 0.0)
            high = constraints.get('max', 20.0)
            params[name] = trial.suggest_float(name, low, high)
        elif param_type.startswith('tuple[int,int]'):
            if constraints.get('odd'):
                low = constraints.get('min', 1)
                high = constraints.get('max', 11)
                candidates = [(i, i) for i in range(low | 1, high + 1, 2)]
                params[name] = trial.suggest_categorical(name, candidates)
            else:
                low = constraints.get('min', 1)
                high = constraints.get('max', 11)
                candidates = [(i, i) for i in range(low, high + 1)]
                params[name] = trial.suggest_categorical(name, candidates)
        else:
            raise ValueError(f"Unsupported type or constraint for param: {name}")

    return params


def create_preprocessing_function(modified_func_str: str, param_values: dict):
    
    for k, v in param_values.items():
        modified_func_str = modified_func_str.replace(str(k), repr(v))
    full_code = "\nimport cv2\ncv = cv2\nimport numpy\nnp = numpy\nfrom src.data_io import ImageData\n" + modified_func_str
    
    local_context = {}
    # Execute preprocessing function in local context
    exec(full_code, local_context, local_context)
    preprocess_func = local_context['preprocess_images']

    return preprocess_func

def evaluate_pipeline(preprocess_func: callable, task: str, data_path: str, **kwargs):
    
    if task == "spot_detection":
        from src.spot_detection import DeepcellSpotsDetector

        detector = DeepcellSpotsDetector()
        spots_data = np.load(f"{data_path}", allow_pickle=True)

        # --- Prepare ImageData ---
        batch_size = spots_data['X'].shape[0]
        images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])    

        processed_img = preprocess_func(images) 

        pred = detector.predict(processed_img)

        metrics = detector.evaluate(pred, spots_data['y'])

        return metrics
    elif task == "cellpose_segmentation":
        from src.cellpose_segmentation import CellposeTool
        segmenter = CellposeTool(model_name="cyto3", device=kwargs.get('gpu_id'))
        # raw_images, gt_masks = segmenter.loadData(data_path)
        raw_images, gt_masks = segmenter.loadCombinedDataset(data_path, kwargs.get('dataset_size'))

        images = ImageData(raw=raw_images, batch_size=kwargs.get('batch_size'), image_ids=[i for i in range(len(raw_images))])

        processed_img = preprocess_func(images) 
        pred_masks = segmenter.predict(processed_img, batch_size=images.batch_size)
        overall_metrics = segmenter.evaluate(pred_masks, gt_masks)
        return overall_metrics
    elif task == "medSAM_segmentation":
        from src.medsam_segmentation import MedSAMTool

        segmenter = MedSAMTool(gpu_id=kwargs.get('gpu_id'), checkpoint_path=kwargs.get('checkpoint_path'))
        raw_images, boxes, masks = segmenter.loadData(data_path)

        # --- Prepare ImageData ---
        batch_size = 8
        images = ImageData(raw=raw_images,
                    batch_size=batch_size,
                    image_ids=[i for i in range(len(raw_images))],
                    masks=masks,
                    predicted_masks=masks)
        
        images = preprocess_func(images)
        
        # --- Run Segmenter ---
        pred_masks = segmenter.predict(images, boxes, used_for_baseline=False)

        overall_metrics = segmenter.evaluate(pred_masks, images.masks)
            
        return overall_metrics

def make_objective(modified_func_str, param_info, task, data_path, kwargs):
    
    if task == "spot_detection":
        metric = 'f1_score'
    elif task == "cellpose_segmentation":
        metric = 'average_precision'
    elif task == "medSAM_segmentation":
        metric = 'dsc_metric'
        
    def objective(trial):
        param_values = define_optuna_search_space(trial, param_info)
        preprocess_func = create_preprocessing_function(modified_func_str, param_values)
        
        score = evaluate_pipeline(preprocess_func, task, data_path, **kwargs)
        
        return score[metric]
    return objective

def save_to_function_bank(func_str: str, metrics: dict, function_bank_path: str, time: float):
    '''
    Save the function string and metrics to a JSON file.

    Args:
        func_str (str): The function string to save.
        metrics (dict): The metrics dictionary to save.
        function_bank_path (str): Path to the function bank JSON file.
        time (float): Time taken for the optimization.
    '''
    
    with open(function_bank_path, 'r') as f:
        function_bank = json.load(f)

    function_bank.append({
        "preprocessing_function": func_str,
        "overall_metrics": metrics,
        "optimization_time": time,
    })

    with open(function_bank_path, 'w') as f:
        json.dump(function_bank, f, indent=4)

def hyperparameter_search(func_str, task: str, data_path: str, log_path: str, n_trials: int = 15, **kwargs):
    '''
    Perform hyperparameter search.

    Args:
        func_str (str): The function string to optimize.
        data_path (str): Path to the data.
        task (str): The task to perform (e.g., "spot_detection").
        log_path (str): Path to the log file.
        n_trials (int): Number of trials for the optimization.
    Returns:
        output (tuple[str, dict]): Tuple of the optimal function string and the metrics dictionary.
    '''
    
    logger = None
    print(f"Setting up logging to file: {log_path}")
    try:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ],
            force=True
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging configured successfully.")
    except Exception as log_e:
        print(f"Error configuring logging: {log_e}. Using basic print statements.")
        class PrintLogger:
            def info(self, msg, *args): print("INFO: " + (msg % args if args else msg))
            def warning(self, msg, *args): print("WARNING: " + (msg % args if args else msg))
            def error(self, msg, *args): print("ERROR: " + (msg % args if args else msg))
            def exception(self, msg, *args): print("EXCEPTION: " + (msg % args if args else msg) + f"\\n{traceback.format_exc()}")
        logger = PrintLogger()
    
    optuna.logging.enable_propagation() 
    optuna.logging.disable_default_handler()
    
    code, params, orig_values = transform_opencv_constants(func_str)
    
    logger.info("Transformed function string:")
    logger.info(code)
    logger.info(f"Parameters extracted: {params}")
    logger.info(f"Original values: {orig_values}")
    
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial(orig_values)
    study_objective = make_objective(code, params, task, data_path, kwargs)
    study.optimize(study_objective, n_trials=n_trials, catch=(cv.error, TypeError, ValueError, SyntaxError))

    best_params = study.best_params

    
    metrics = evaluate_pipeline(create_preprocessing_function(code, best_params), task, data_path, **kwargs)

    for p in best_params:
        code = code.replace(p, str(best_params[p]))

    return code, metrics