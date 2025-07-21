""" Baseline AutoML Search """
import argparse
import os
from datetime import timedelta

import cv2 as cv
import numpy as np
import functools
import optuna
from optuna.samplers import TPESampler, RandomSampler
import wandb
import json
import torch

from src.data_io import ImageData
from src.utils import set_gpu_device


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)


def dtype_guard(fn):
    @functools.wraps(fn)
    def wrapped(img, *args, **kwargs):
        # ensure uint8 inputs for OpenCV if needed
        if img.dtype not in (np.uint8, np.float32, np.float64):
            img = to_uint8(img)
        return fn(img, *args, **kwargs)

    return wrapped


def grayify(img):
    u8 = to_uint8(img)
    if u8.ndim == 2:
        return u8
    C = u8.shape[2]
    if C == 3:
        return cv.cvtColor(u8, cv.COLOR_BGR2GRAY)
    if C == 4:
        return cv.cvtColor(u8, cv.COLOR_BGRA2GRAY)
    return u8.mean(axis=2).astype(np.uint8)


def normalize_u8(arr):
    return cv.normalize(arr, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def build_kernel(trial, prefix, allow_gabor=True):
    kinds = ["structuring", "gaussian"] + (['gabor'] if allow_gabor else [])
    kt = trial.suggest_categorical(f"{prefix}_kernel", kinds)
    k = trial.suggest_int(f"{prefix}_k", 3, 11, step=2)
    if kt == "structuring":
        shape = trial.suggest_categorical(f"{prefix}_shape", [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS])
        return cv.getStructuringElement(shape, (k, k))
    if kt == "gaussian":
        sigma = trial.suggest_float(f"{prefix}_sigma", 0.1, 3.0)
        g1d = cv.getGaussianKernel(k, sigma)
        return g1d @ g1d.T
    if kt == "gabor":
        sigma = trial.suggest_float(f"{prefix}_sigma", 1.0, 5.0)
        theta = trial.suggest_float(f"{prefix}_theta", 0, np.pi)
        lam = trial.suggest_float(f"{prefix}_lambd", 4.0, 10.0)
        gamma = trial.suggest_float(f"{prefix}_gamma", 0.5, 1.0)
        return cv.getGaborKernel((k, k), sigma, theta, lam, gamma)
    raise ValueError(f"Unknown kernel type {kt}")


def build_linear(trial):
    op = trial.suggest_categorical("linear_op", [
        "bilateralFilter", "filter2D", "blur",
        "boxFilter", "GaussianBlur", "medianBlur",
    ])
    border = trial.suggest_categorical("linear_border", [cv.BORDER_DEFAULT, cv.BORDER_REFLECT, cv.BORDER_REPLICATE]) # add border type
    if op == "bilateralFilter":
        d = trial.suggest_int("linear_bilateral_d", 3, 9, step=2)
        sc = trial.suggest_int("linear_bilateral_sigmaColor", 10, 150)
        ss = trial.suggest_int("linear_bilateral_sigmaSpace", 10, 150)

        @dtype_guard
        def fn(img):
            return cv.bilateralFilter(to_uint8(img), d, sc, ss, borderType=border)

        return fn
    if op == "filter2D":
        kernel = build_kernel(trial, "linear_2d")

        @dtype_guard
        def fn(img):
            return cv.filter2D(img, -1, kernel, borderType=border)

        return fn
    if op == "blur":
        k = trial.suggest_int("linear_blur_k", 3, 11, step=2)

        @dtype_guard
        def fn(img):
            return cv.blur(img, (k, k), borderType=border)

        return fn
    if op == "boxFilter":
        k = trial.suggest_int("linear_box_k", 3, 11, step=2)
        norm = trial.suggest_categorical("linear_box_norm", [True, False])

        @dtype_guard
        def fn(img):
            return cv.boxFilter(img, -1, (k, k), normalize=norm, borderType=border)

        return fn
    if op == "GaussianBlur":
        k = trial.suggest_int("linear_gauss_k", 3, 11, step=2)
        sg = trial.suggest_float("linear_gauss_sigma", 0.1, 3.0)

        @dtype_guard
        def fn(img):
            return cv.GaussianBlur(img, (k, k), sg, borderType=border)

        return fn
    if op == "medianBlur":
        k = trial.suggest_int("linear_med_k", 3, 11, step=2)

        @dtype_guard
        def fn(img):
            return cv.medianBlur(to_uint8(img), k)

        return fn
    raise ValueError(f"Unknown linear operation: {op}")


def build_morphology(trial):
    op_name = trial.suggest_categorical("morph_op", [
        "erode", "dilate", "open", "close", "gradient", "tophat", "blackhat",
    ])
    ksize = trial.suggest_int("morph_k", 3, 11, step=2)
    shape = trial.suggest_categorical("morph_shape", [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS])
    kernel = cv.getStructuringElement(shape, (ksize, ksize))
    iters = trial.suggest_int("morph_iters", 1, 3)
    flag = {
        "erode": cv.MORPH_ERODE,
        "dilate": cv.MORPH_DILATE,
        "open": cv.MORPH_OPEN,
        "close": cv.MORPH_CLOSE,
        "gradient": cv.MORPH_GRADIENT,
        "tophat": cv.MORPH_TOPHAT,
        "blackhat": cv.MORPH_BLACKHAT,
    }[op_name]

    @dtype_guard
    def fn(img):
        return cv.morphologyEx(img, flag, kernel, iterations=iters)

    return fn


def build_derivative(trial):
    op = trial.suggest_categorical("deriv_op", ["Sobel", "Scharr", "Laplacian"])
    if op == "Sobel":
        dx = trial.suggest_int("sobel_dx", 0, 1)
        dy = 1 - dx
        k = trial.suggest_int("sobel_k", 1, 7, step=2)

        @dtype_guard
        def fn(img):
            sb = cv.Sobel(img, cv.CV_32F, dx, dy, ksize=k)
            return normalize_u8(sb)

        return fn
    if op == "Scharr":
        dx = trial.suggest_int("sch_dx", 0, 1)
        dy = 1 - dx

        @dtype_guard
        def fn(img):
            sc = cv.Scharr(img, cv.CV_32F, dx, dy)
            return normalize_u8(sc)

        return fn
    if op == "Laplacian":
        k = trial.suggest_int("lap_k", 1, 7, step=2)

        @dtype_guard
        def fn(img):
            lp = cv.Laplacian(img, cv.CV_32F, ksize=k)
            return normalize_u8(lp)

        return fn
    raise ValueError(f"Unknown derivative operation: {op}")


def build_pyramids(trial):
    sp = trial.suggest_float("ms_sp", 3.0, 20.0)
    sr = trial.suggest_float("ms_sr", 3.0, 20.0)

    def fn(img):
        u8 = to_uint8(img)
        if u8.ndim == 2:
            bgr = cv.cvtColor(u8, cv.COLOR_GRAY2BGR)
        else:
            c = u8.shape[2]
            if c == 1:
                bgr = cv.cvtColor(u8[..., 0], cv.COLOR_GRAY2BGR)
            elif c == 3:
                bgr = u8
            elif c == 4:
                bgr = cv.cvtColor(u8, cv.COLOR_BGRA2BGR)
            else:
                gray = grayify(u8)
                bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        ms = cv.pyrMeanShiftFiltering(bgr, sp, sr)
        gray = cv.cvtColor(ms, cv.COLOR_BGR2GRAY)
        return gray

    return dtype_guard(fn)


def build_threshold(trial):
    op = trial.suggest_categorical("thr_op", ["threshold", "adaptiveThreshold", "equalizeHist", "CLAHE"])
    if op == "threshold":
        thr = trial.suggest_int("thr_thr", 0, 255)
        mval = trial.suggest_int("thr_mval", 0, 255)
        typ = trial.suggest_categorical("thr_type", [
            cv.THRESH_BINARY,
            cv.THRESH_BINARY_INV,
            cv.THRESH_TRUNC,
            cv.THRESH_TOZERO,
            cv.THRESH_BINARY | cv.THRESH_OTSU,
            cv.THRESH_BINARY_INV | cv.THRESH_OTSU
        ])
        @dtype_guard
        def fn(img):
            _, dst = cv.threshold(grayify(img), thr, mval, typ)
            return dst

        return fn
    if op == "adaptiveThreshold":
        maxV = trial.suggest_int("at_maxV", 100, 255)
        aM = trial.suggest_categorical("at_adapt", [cv.ADAPTIVE_THRESH_MEAN_C, cv.ADAPTIVE_THRESH_GAUSSIAN_C])
        tT = trial.suggest_categorical("at_type", [cv.THRESH_BINARY, cv.THRESH_BINARY_INV])
        bS = trial.suggest_int("at_blk", 3, 15, step=2)
        C = trial.suggest_float("at_C", 0, 10)

        @dtype_guard
        def fn(img): return cv.adaptiveThreshold(grayify(img), maxV, aM, tT, bS, int(C))

        return fn
    if op == "equalizeHist":
        @dtype_guard
        def fn(img): return cv.equalizeHist(grayify(img))

        return fn
    if op == "CLAHE":
        clip = trial.suggest_float("clahe_clip", 1.0, 10.0)
        gx = trial.suggest_int("clahe_gx", 4, 16, step=4)
        gy = trial.suggest_int("clahe_gy", 4, 16, step=4)
        clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(gx, gy))

        @dtype_guard
        def fn(img): return clahe.apply(grayify(img))

        return fn
    raise ValueError(f"Unknown threshold op: {op}")


def build_denoise(trial):
    op = trial.suggest_categorical("dn_op", ["fastDenoiseGray", "fastDenoiseColor"])
    if op == "fastDenoiseGray":
        h = trial.suggest_float("fdg_h", 3, 10)
        tW = trial.suggest_int("fdg_tw", 7, 21, step=2)
        sW = trial.suggest_int("fdg_sw", 21, 61, step=2)

        @dtype_guard
        def fn(img): return cv.fastNlMeansDenoising(grayify(img), None, h, tW, sW)

        return fn
    if op == "fastDenoiseColor":
        h = trial.suggest_float("fdc_h", 3, 10)
        hc = trial.suggest_float("fdc_hc", 3, 10)
        tW = trial.suggest_int("fdc_tw", 7, 21, step=2)
        sW = trial.suggest_int("fdc_sw", 21, 61, step=2)

        @dtype_guard
        def fn(img):
            u8 = to_uint8(img)
            if u8.ndim != 3 or u8.shape[2] != 3:
                return cv.fastNlMeansDenoising(grayify(u8), None, h, tW, sW)
            return cv.fastNlMeansDenoisingColored(u8, None, h, hc, tW, sW)

        return fn
    raise ValueError(f"Unknown denoise op: {op}")


def build_color(trial):
    op = trial.suggest_categorical("color_op", ["cvtColor", "mergeChannels"])
    if op == "cvtColor":
        code = trial.suggest_categorical("ccode", [
            cv.COLOR_BGR2GRAY,
            cv.COLOR_BGR2HSV,
            cv.COLOR_BGR2LAB,
            cv.COLOR_HSV2RGB,
            cv.COLOR_LAB2BGR,
            cv.COLOR_LAB2RGB,
            cv.COLOR_RGB2HSV,
            cv.COLOR_RGB2LAB,
            cv.COLOR_RGB2YCrCb,
            cv.COLOR_RGB2YUV,
            cv.COLOR_YCrCb2RGB,
            cv.COLOR_YUV2RGB,
        ])

        @dtype_guard
        def fn(img):
            u8 = to_uint8(img)
            if u8.ndim == 3 and u8.shape[2] == 1:
                gray = u8[..., 0]
            elif u8.ndim == 2:
                gray = u8
            else:
                gray = None
            if gray is not None:
                if code in (cv.COLOR_BGR2GRAY, cv.COLOR_BGRA2GRAY):
                    return gray
                u8 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            if u8.ndim == 3 and u8.shape[2] == 4 and code != cv.COLOR_BGRA2GRAY:
                u8 = cv.cvtColor(u8, cv.COLOR_BGRA2BGR)
            return cv.cvtColor(u8, code)

        return fn

    if op == "mergeChannels":
        @dtype_guard
        def fn(img):
            if img.ndim < 3 or img.shape[2] < 3:
                return to_uint8(img)
            ch = cv.split(to_uint8(img))
            channels = [ch[2], ch[1], ch[0]]
            if len(ch) > 3:
                channels.extend(ch[3:])
            return cv.merge(channels)

        return fn
    else:
        raise ValueError(f"Unknown color op: {op}")


def build_spatial(trial):
    op = trial.suggest_categorical("sp_op", ["Canny", "flip"])
    if op == "Canny":
        t1 = trial.suggest_int("canny_t1", 50, 150)
        t2 = trial.suggest_int("canny_t2", 100, 250)

        @dtype_guard
        def fn(img): return cv.Canny(grayify(img), t1, t2)

        return fn
    if op == "flip":
        code = trial.suggest_categorical("flip_code", [0, 1, -1])

        @dtype_guard
        def fn(img):
            return cv.flip(img, code)

        return fn

    else:
        raise ValueError(f"Unknown spatial op: {op}")


def build_arithmetic(trial):
    op = trial.suggest_categorical("arith_op", ["add", "subtract", "magnitude", "convertScaleAbs"])
    if op == "add":
        val = trial.suggest_int("arith_add_val", 0, 100)
        @dtype_guard
        def fn(img):
            return cv.add(img, val)
        return fn
    if op == "subtract":
        val = trial.suggest_int("arith_sub_val", 0, 100)
        @dtype_guard
        def fn(img):
            return cv.subtract(img, val)
        return fn
    if op == "magnitude":
        @dtype_guard
        def fn(img):
            u8 = to_uint8(img)
            ch = cv.split(u8)
            if len(ch) < 2:
                return u8
            mag = cv.magnitude(ch[0].astype(np.float32), ch[1].astype(np.float32))
            return normalize_u8(mag)
        return fn
    if op == "convertScaleAbs":
        alpha = trial.suggest_float("arith_csa_alpha", 0.5, 2.0)
        beta = trial.suggest_int("arith_csa_beta", 0, 100)
        @dtype_guard
        def fn(img):
            return cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        return fn
    raise ValueError(f"Unknown arithmetic operation: {op}")


def preprocess_op_from_trial(trial):
    method = trial.suggest_categorical("method", [
        "linear", "morphology", "derivative", "pyramids", "threshold",
         "denoise", "color", "spatial", "arithmetic"
    ])
    builders = {
        "linear": build_linear,
        "morphology": build_morphology,
        "derivative": build_derivative,
        "pyramids": build_pyramids,
        "threshold": build_threshold,
        "denoise": build_denoise,
        "color": build_color,
        "spatial": build_spatial,
        "arithmetic": build_arithmetic,
    }
    return builders[method](trial)


class PrefixedTrial:
    def __init__(self, trial, prefix):
        self._trial = trial
        self._prefix = prefix

    def suggest_categorical(self, name, choices):
        return self._trial.suggest_categorical(self._prefix + name, choices)

    def suggest_int(self, name, low, high, **kw):
        return self._trial.suggest_int(self._prefix + name, low, high, **kw)

    def suggest_float(self, name, low, high, **kw):
        return self._trial.suggest_float(self._prefix + name, low, high, **kw)


def build_pipeline_from_trial(trial, max_stages):
    n_stages = trial.suggest_int("n_stages", 2, max_stages)
    fns = [preprocess_op_from_trial(PrefixedTrial(trial, f"stage{i}_")) for i in range(n_stages)]

    def pipeline(img):
        orig_ndim = img.ndim
        orig_channels = img.shape[2] if orig_ndim == 3 else 1
        for i, fn in enumerate(fns):
            try:
                img = fn(img)
            except Exception as e:
                print(f"Error in stage {i}: {e}")
                continue
            if img.ndim == 2:
                img = img[..., None]
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.5:
            img = img / 255.0
        curr_ch = img.shape[2] if img.ndim == 3 else 1
        if curr_ch != orig_channels:
            if orig_channels == 1 and curr_ch > 1:
                gray = cv.cvtColor((img * 255).astype(np.uint8).squeeze(), cv.COLOR_BGR2GRAY)
                img = gray.astype(np.float32) / 255.0
                img = img[..., None]
            elif orig_channels == 3 and curr_ch == 1:
                bgr = cv.cvtColor((img.squeeze() * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
                img = bgr.astype(np.float32) / 255.0
            elif orig_channels == 4 and curr_ch == 3:
                alpha = np.ones(img.shape[:2] + (1,), dtype=np.float32)
                img = np.concatenate([img, alpha], axis=2)
        if orig_ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze()
        return img

    return pipeline


def preprocess_batch(images: ImageData, fn) -> ImageData:
    return ImageData(raw=[fn(img) for img in images.raw], batch_size=images.batch_size)


def build_objective(experiment_name, trials_table, val_data_path, checkpoint_path, max_stages, gpu_id):
    if experiment_name == "spot_detection":
        from src.spot_detection import DeepcellSpotsDetector

        spot_detector = DeepcellSpotsDetector()

        def spot_objective(trial):
            spots_data_val = np.load(val_data_path, allow_pickle=True)
            val_images = ImageData(raw=spots_data_val['X'], batch_size=spots_data_val['X'].shape[0],
                                   image_ids=[i for i in range(spots_data_val['X'].shape[0])])
            val_y = spots_data_val['y']
            preprocess_fn = build_pipeline_from_trial(trial, max_stages)
            val_processed = preprocess_batch(val_images, preprocess_fn)
            val_pred = spot_detector.predict(val_processed)
            val_metrics = spot_detector.evaluate(val_pred, val_y)
            val_total = val_metrics["class_loss"] + val_metrics["regress_loss"]
            wandb_metrics = {
                'validate/total_loss': val_total,
                'validate/class_loss': val_metrics["class_loss"],
                'validate/regress_loss': val_metrics["regress_loss"],
                'validate/f1_score': val_metrics["f1_score"]
            }
            table_metrics = [val_total, val_metrics["f1_score"]]
            wandb_log_metrics(trial, wandb_metrics, table_metrics, trials_table)
            return val_metrics["f1_score"]

        return spot_objective

    elif experiment_name == "cellpose_segmentation":
        from src.cellpose_segmentation import CellposeTool

        cell_segmenter = CellposeTool(model_name="cyto3", device=gpu_id)

        def cellpose_objective(trial):
            val_dataset_size = 100
            batch_size = 16

            val_raw_images, val_gt_masks, val_image_sources = cell_segmenter.loadCombinedDataset(val_data_path, val_dataset_size)
            val_images = ImageData(raw=val_raw_images, batch_size=batch_size, image_ids=val_image_sources)
            preprocess_fn = build_pipeline_from_trial(trial, max_stages)
            val_processed = preprocess_batch(val_images, preprocess_fn)
            val_pred_masks = cell_segmenter.predict(val_processed, batch_size=val_processed.batch_size)
            val_new_images = ImageData(raw=val_raw_images, batch_size=batch_size, image_ids=val_image_sources, masks=val_gt_masks,
                                   predicted_masks=val_pred_masks)
            val_avg_precision = cell_segmenter.evaluateDisaggregated(val_new_images)["average_precision"]
            wandb_metrics = {
                'validate/average_precision': val_avg_precision,
            }
            table_metrics = list(wandb_metrics.values())
            wandb_log_metrics(trial, wandb_metrics, table_metrics, trials_table)
            return val_avg_precision

        return cellpose_objective

    elif experiment_name == "medSAM_segmentation":
        from src.medsam_segmentation import MedSAMTool

        medsam_segmenter = MedSAMTool(gpu_id=gpu_id, checkpoint_path=checkpoint_path)

        def medsam_objective(trial):
            batch_size = 8

            val_raw_images, val_boxes, val_masks = medsam_segmenter.loadData(val_data_path)
            val_images = ImageData(raw=val_raw_images,
                               batch_size=batch_size,
                               image_ids=[i for i in range(len(val_raw_images))],
                               masks=val_masks,
                               predicted_masks=val_masks)
            preprocess_fn = build_pipeline_from_trial(trial, max_stages)
            val_processed = preprocess_batch(val_images, preprocess_fn)
            val_pred = medsam_segmenter.predict(val_processed, val_boxes, used_for_baseline=False)
            val_metrics = medsam_segmenter.evaluate(val_pred, val_images.masks)
            val_total = val_metrics['dsc_metric'] + val_metrics['nsd_metric']
            wandb_metrics = {
                'validate/total_dsc+nsd': val_total,
                'validate/dsc_metric': val_metrics["dsc_metric"],
                'validate/nsd_metric': val_metrics["nsd_metric"],
            }
            table_metrics = [val_total]
            wandb_log_metrics(trial, wandb_metrics, table_metrics, trials_table)
            return val_total

        return medsam_objective
    else:
        raise ValueError(f"Unknown experiment name {experiment_name}")


def wandb_log_metrics(trial, wandb_metrics, table_metrics, trials_table):
    pipeline_config = [f"{i}_{trial.params[f'stage{i}_method']}" for i in range(trial.params['n_stages'])]
    trial.set_user_attr('pipeline', pipeline_config)

    for metric, value in wandb_metrics.items():
        trial.set_user_attr(metric, value)

    pipeline_str = ",".join(pipeline_config)

    trials_table.add_data(
        trial.number,
        trial.params['n_stages'],
        pipeline_str,
        *table_metrics,
        json.dumps(trial.params)
    )

    log_trials_table = wandb.Table(
        columns=trials_table.columns, data=trials_table.data
    )

    wandb.log({
        'trial': trial.number,
        **wandb_metrics,
        'trials': log_trials_table
    })


class FakeTrial:
    def __init__(self, params):
        self.params = params
    def suggest_categorical(self, name, choices):
        return self.params[name]
    def suggest_int(self, name, low, high, **kw):
        return self.params[name]
    def suggest_float(self, name, low, high, **kw):
        return self.params[name]


def _json_fallback(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    return str(obj)


def save_study_to_json(study, filename):

    study_dict = {
        "best_trial": {
            "number": study.best_trial.number,
            "duration": study.best_trial.duration,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs
        },
        "trials": []
    }

    for trial in study.trials:
        if trial.state.is_finished():
            trial_dict = {
                "number": trial.number,
                "duration": trial.duration,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "state": trial.state.name
            }
            study_dict["trials"].append(trial_dict)

    with open(filename, 'w') as f:
        json.dump(study_dict, f, indent=2, default=_json_fallback)

    print(f"Study saved to {filename}")

    return study_dict


def eval_k_trials(study_dict, args, save_name, k=10):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cv.setRNGSeed(args.seed)

    sorted_trials = sorted(study_dict["trials"], key=lambda t: t["value"], reverse=True)
    top_k_trials = sorted_trials[:k]

    for rank, trial in enumerate(top_k_trials, start=1):
        trial["rank"] = rank

        params = trial["params"]

        preprocess_fn = build_pipeline_from_trial(FakeTrial(params), args.max_stages)

        print(f"Evaluating Trial Ranked {rank}")

        if args.experiment_name == "spot_detection":
            from src.spot_detection import DeepcellSpotsDetector

            spot_detector = DeepcellSpotsDetector()

            val_data_dir = os.path.dirname(args.val_data_path)

            spots_data_test = np.load(os.path.join(val_data_dir, "test.npz"), allow_pickle=True)

            test_images = ImageData(raw=spots_data_test['X'], batch_size=spots_data_test['X'].shape[0],
                                    image_ids=[i for i in range(spots_data_test['X'].shape[0])])
            test_y = spots_data_test['y']

            processed_test = preprocess_batch(test_images, preprocess_fn)
            pred_test = spot_detector.predict(processed_test)
            test_metrics = spot_detector.evaluate(pred_test, test_y)

        elif args.experiment_name == "cellpose_segmentation":
            from src.cellpose_segmentation import CellposeTool

            cell_segmenter = CellposeTool(model_name="cyto3", device=0)
            test_dataset_size = 808
            test_data_path = os.path.join(os.path.dirname(os.path.normpath(args.val_data_path)), "test_set", "")
            batch_size = 16
            test_raw_images, test_gt_masks, val_image_sources = cell_segmenter.loadCombinedDataset(test_data_path,
                                                                                                   test_dataset_size)
            test_images = ImageData(raw=test_raw_images, batch_size=batch_size, image_ids=val_image_sources)
            test_processed = preprocess_batch(test_images, preprocess_fn)
            test_pred_masks = cell_segmenter.predict(test_processed, batch_size=test_processed.batch_size)
            test_new_images = ImageData(raw=test_raw_images, batch_size=batch_size, image_ids=val_image_sources,
                                        masks=test_gt_masks,
                                        predicted_masks=test_pred_masks)
            test_metrics = cell_segmenter.evaluateDisaggregated(test_new_images)

        elif args.experiment_name == "medSAM_segmentation":
            from src.medsam_segmentation import MedSAMTool

            medsam_segmenter = MedSAMTool(gpu_id=0, checkpoint_path=args.checkpoint_path)
            batch_size = 4

            val_data_dir = args.val_data_path
            test_data_path = val_data_dir.replace('_val.pkl', '_test.pkl')

            test_raw_images, test_boxes, test_masks = medsam_segmenter.loadData(test_data_path)
            test_images = ImageData(raw=test_raw_images,
                                    batch_size=batch_size,
                                    image_ids=[i for i in range(len(test_raw_images))],
                                    masks=test_masks,
                                    predicted_masks=test_masks)
            test_processed = preprocess_batch(test_images, preprocess_fn)
            test_pred = medsam_segmenter.predict(test_processed, test_boxes, used_for_baseline=False)
            test_metrics = medsam_segmenter.evaluate(test_pred, test_images.masks)
            test_metrics["total_dsc+nsd"] = test_metrics['dsc_metric'] + test_metrics['nsd_metric']

        else:
            raise ValueError(f"Unknown task: {args.experiment_name}")

            # Attach computed metrics to this trial
        trial["test_metrics"] = test_metrics

    # Save top k trials with their metrics to a JSON file
    output_file = f"{save_name}_top_{k}.json"
    with open(output_file, "w") as out_f:
        json.dump({"trials": top_k_trials}, out_f, indent=2, default=_json_fallback)
    print(f"Saved top {k} trials with metrics to {output_file}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna preprocessing search for biomedical image processing tasks')
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=["spot_detection", "cellpose_segmentation", "medSAM_segmentation"],
        help="Name of the experiment. Must be one of: spot_detection, cellpose_segmentation, medSAM_segmentation"
    )
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data; will be used to infer the test data path")
    parser.add_argument('--n_trials', type=int, default=1200,
                        help='Number of trials for optimization')
    parser.add_argument('--checkpoint_path', type=str, required=False, help="Checkpoint path for medSAM")
    parser.add_argument('--max_stages', type=int, default=4, help='Maximum number of stages in the pipeline. Minimum is set to 2')
    parser.add_argument("--opt_direction", type=str, default=None,
                        choices=["minimize", "maximize"],
                        help="Direction of optimization. Must be one of: minimize, maximize")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument('--wandb', type=str, default='disabled', help="Whether to enabled wandb logging; disabled by default")
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random'], help="Sampler for Optuna")
    args = parser.parse_args()

    set_gpu_device(args.gpu_id)

    N_TRIALS = args.n_trials
    MAX_STAGES = args.max_stages
    EXPERIMENT_NAME = args.experiment_name
    SEED = args.seed
    SAMPLER = args.sampler

    opt_direction = {
        "spot_detection": "maximize",
        "cellpose_segmentation": "maximize",
        "medSAM_segmentation": "maximize"
    }

    if args.opt_direction is not None:
        opt_direction[EXPERIMENT_NAME] = args.opt_direction

    wandb.init(mode=args.wandb, project=f"{EXPERIMENT_NAME}_optuna",
               config={"n_trials": N_TRIALS, "sampler": SAMPLER, "max_stages": MAX_STAGES, "opt_direction": opt_direction[EXPERIMENT_NAME], "seed": SEED, "val_data_path": args.val_data_path})

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    cv.setRNGSeed(SEED)

    study = optuna.create_study(
        direction=opt_direction[EXPERIMENT_NAME],
        sampler=TPESampler(seed=SEED) if SAMPLER == "tpe" else RandomSampler(seed=SEED),
    )

    save_file_suffix = f"_{SAMPLER}"

    if EXPERIMENT_NAME == "spot_detection":
        task_wandb_table = wandb.Table(
            columns=["trial", "n_stages", "pipeline", "val_total_loss", "val_f1", "params"])
    elif EXPERIMENT_NAME == "cellpose_segmentation":
        task_wandb_table = wandb.Table(
            columns=["trial", "n_stages", "pipeline", "val_avg_precision", "params"])
    elif EXPERIMENT_NAME == "medSAM_segmentation":
        task_wandb_table = wandb.Table(
            columns=["trial", "n_stages", "pipeline", "val_total_dsc+nsd", "params"])
        save_file_suffix = f"_{os.path.splitext(os.path.basename(args.val_data_path))[0]}_{save_file_suffix}"
    else:
        raise ValueError(f"Unknown experiment name {EXPERIMENT_NAME}")

    objective = build_objective(EXPERIMENT_NAME, task_wandb_table, args.val_data_path, args.checkpoint_path, MAX_STAGES, args.gpu_id)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        catch=(cv.error, TypeError, ValueError),
    )

    save_name = f"{EXPERIMENT_NAME}_trials{save_file_suffix}"

    study_dict = save_study_to_json(study, f"{save_name}.json")

    print("\nBest trial:")
    bt = study.best_trial
    print(f"  Value:    {bt.value:.4f}")
    print(f"  Pipeline: {bt.user_attrs['pipeline']}")
    print("\nParameters:")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")

    eval_k_trials(study_dict, args, save_name)