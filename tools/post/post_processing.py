import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2])) 

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from models.yolo_v8 import YOLOModel
from src.evaluate import evaluate_detection
from src.data_preprocessing import apply_custom_preprocessing

__all__ = [
    "sweep_thresholds_and_evaluate",
    "predict_to_yolo_labels",
]

# -----------------------------------------------------------------------------
# Helper – automatically figure out where images / labels live inside test_dir
# -----------------------------------------------------------------------------

def _auto_paths(test_dir: str) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) under *test_dir*.

    - Prefers ./images_processed over ./images
    - Prefers ./labels_yolo over ./labels
    """
    tdir = Path(test_dir)

    # images directory --------------------------------------------------------
    img_dir = tdir / "images_processed"
    if not img_dir.is_dir():  # fallback
        img_dir = tdir / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find images or images_processed inside {test_dir!r}"
        )

    # labels directory --------------------------------------------------------
    lbl_dir = tdir / "labels_yolo"
    if not lbl_dir.is_dir():
        lbl_dir = tdir / "labels"
    if not lbl_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find labels_yolo or labels inside {test_dir!r}"
        )

    return img_dir, lbl_dir


# -----------------------------------------------------------------------------
# Prediction helper – save YOLO-format *.txt for every image in img_dir
# -----------------------------------------------------------------------------

def predict_to_yolo_labels(
    model: YOLOModel,
    img_dir: Path,
    out_dir: Path,
    conf_thres: float = 0.25,
    preprocess: bool = True,
    verbose: bool = False,
):
    """Run *model* over all images in *img_dir* and dump YOLO‑style label files.

    Any existing directory *out_dir* will be **cleared**.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # wipe existing content
    for f in out_dir.glob("*.txt"):
        f.unlink()

    img_paths = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
    )

    for idx, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            if verbose:
                print(f"[WARN] failed reading {img_path}")
            continue

        if preprocess:
            img_bgr = apply_custom_preprocessing(img_bgr, use_normalization=False)

        # model.predict expects BGR np.array; conf filtering is handled below
        boxes = model.predict(img_bgr, conf_thresh=conf_thres)

        # write labels --------------------------------------------------------
        if boxes:
            h, w = img_bgr.shape[:2]
            label_lines = []
            for x1, y1, x2, y2, score, cls_id in boxes:
                # skip below threshold (model already filters but double‑check)
                if score < conf_thres:
                    continue
                # convert to YOLO rel format
                x_c = (x1 + x2) / 2 / w
                y_c = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{int(cls_id)} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} {score:.6f}")

            if label_lines:
                label_path = out_dir / (img_path.stem + ".txt")
                label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if verbose and (idx + 1) % 200 == 0:
            print(f"  processed {idx+1}/{len(img_paths)} images…")


# -----------------------------------------------------------------------------
# Public API – sweep thresholds, keep best on mAP@0.5
# -----------------------------------------------------------------------------

def sweep_thresholds_and_evaluate(
    model_weights: str,
    test_dir: str,
    thresholds: List[float] = None,
    device: str = "auto",
    preprocess: bool = True,
    keep_best_pred_dir: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, float]:
    """Evaluate a trained model over *test_dir* for several confidence thresholds.

    Parameters
    ----------
    model_weights : str
        Path to ``best.pt`` (or any YOLOv8 `.pt`).
    test_dir : str
        Folder that contains either ``images``/** or ``images_processed``/**
        *and* ``labels_yolo`` or ``labels``.
    thresholds : list[float], optional
        confidences to try. Defaults to ``[0.05,0.1,0.15,…,0.5]``.
    keep_best_pred_dir : str, optional
        If given, predictions from the best threshold are copied here.
    debug : bool
        Verbose logging & class‑level metrics from evaluate_detection.

    Returns
    -------
    dict
        metrics from the best threshold (mAP@0.5, precision, recall, f1 …)
    """
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(1, 11)]  # 0.05…0.5

    img_dir, gt_dir = _auto_paths(test_dir)

    # temp workspace ---------------------------------------------------------
    work_root = Path(tempfile.mkdtemp(prefix="pred_sweep_"))

    try:
        # load model ---------------------------------------------------------
        model = YOLOModel(device=device)
        model.load(model_weights)

        best_map = -1.0
        best_metrics: Dict[str, float] = {}
        best_thr = None
        best_pred_dir = None

        for thr in thresholds:
            pred_dir = work_root / f"pred_{thr:.2f}"
            predict_to_yolo_labels(
                model,
                img_dir,
                pred_dir,
                conf_thres=thr,
                preprocess=preprocess,
                verbose=debug,
            )

            metrics = evaluate_detection(str(gt_dir), str(pred_dir), debug=False)
            map50 = metrics.get("mAP@0.5", 0.0)
            if debug:
                print(f"thr={thr:.2f}  mAP@0.5={map50:.4f}  f1={metrics['f1']:.4f}")

            if map50 > best_map:
                best_map = map50
                best_metrics = metrics
                best_thr = thr
                best_pred_dir = pred_dir

        # optional: persist best predictions --------------------------------
        if keep_best_pred_dir:
            keep_path = Path(keep_best_pred_dir)
            if keep_path.exists():
                shutil.rmtree(keep_path)
            shutil.copytree(best_pred_dir, keep_path)
            if debug:
                print(f"[INFO] best predictions copied to {keep_path}")

        if debug:
            print("\n============= SUMMARY =============")
            print(f"Best confidence threshold : {best_thr:.2f}")
            for k, v in best_metrics.items():
                if isinstance(v, float):
                    print(f"{k:12}: {v:.4f}")

        return {
            "best_threshold": best_thr,
            **best_metrics,
        }

    finally:
        # clean temp preds (unless user wants to inspect)
        shutil.rmtree(work_root, ignore_errors=True)
