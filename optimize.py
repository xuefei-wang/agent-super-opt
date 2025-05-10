import ast
import types
import inspect
import optuna
import json

import cv2
import numpy as np
from src.data_io import ImageData
from src.spot_detection import DeepcellSpotsDetector
from assets.opencv_arg_rules import OPENCV_ARG_RULES
from prompts.task_prompts import TaskPrompts

class CV2ParamExtractor(ast.NodeTransformer):
    def __init__(self):
        self.param_counter = 0
        self.param_info = []

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
            param_name = f"param_{self.param_counter}"
            constraints = rule.get("constraints", {}) if rule else {}
            self.param_info.append({
                "name": param_name,
                "type": ptype,
                "constraints": constraints
            })
            self.param_counter += 1
            return ast.Name(id=param_name, ctx=ast.Load())
        elif isinstance(node, ast.Tuple):
            elements = []
            for i, el in enumerate(node.elts):
                el_rule = None
                if rule and rule["type"].startswith("tuple") and "int" in rule["type"]:
                    el_rule = {"type": "int", "constraints": rule.get("constraints", {})}
                elements.append(self.replace_constants(el, el_rule))
            return ast.Tuple(elts=elements, ctx=ast.Load())
        return node

def transform_opencv_constants(func_str):
    tree = ast.parse(func_str)
    transformer = CV2ParamExtractor()
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree), transformer.param_info


def define_optuna_search_space(trial, param_info):
    params = {}
    for param in param_info:
        name = param['name']
        param_type = param['type']
        constraints = param['constraints']

        if param_type == 'int':
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

        elif 'enum' in constraints:
            params[name] = trial.suggest_categorical(name, constraints['enum'])

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

def evaluate_pipeline(preprocess_func: callable, task: str, data_path: str):
    
    if task == "spot_detection":
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
        #TODO
        pass
    elif task == "medSAM_segmentation":
        #TODO
        pass

def make_objective(modified_func_str, param_info, task, data_path, metric: str):
    def objective(trial):
        param_values = define_optuna_search_space(trial, param_info)
        preprocess_func = create_preprocessing_function(modified_func_str, param_values)
        
        score = evaluate_pipeline(preprocess_func, task, data_path)
        
        return score[metric]
    return objective


def hyperparameter_search(func_str, task: str, data_path: str, metric: str, n_trials: int = 15):
    '''
    Perform hyperparameter search.

    Args:
        func_str (str): The function string to optimize.
        data_path (str): Path to the data.
        task (str): The task to perform (e.g., "spot_detection").
        metric (str): The metric to optimize.
        n_trials (int): Number of trials for the optimization.
    Returns:
        output (tuple[str, dict]): Tuple of the optimal function string and the metrics dictionary.
    '''
    code, params = transform_opencv_constants(func_str)

    study = optuna.create_study(direction='maximize')
    study_objective = make_objective(code, params, task, data_path, metric)
    study.optimize(study_objective, n_trials=n_trials)

    best_params = study.best_params

    
    metrics = evaluate_pipeline(create_preprocessing_function(code, best_params), task, data_path)

    for p in best_params:
        code = code.replace(p, str(best_params[p]))

    return code, metrics