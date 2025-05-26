#!/usr/bin/env python
"""inspect_labels.py
-------------------------------------
A lightweight utility that scans a directory of YOLO-format ``*.txt`` label
files and prints class-distribution statistics – handy for spotting class
imbalance or faulty label conversion.

Usage
-----
$ python inspect_labels.py path/to/labels_yolo             # human readable
$ python inspect_labels.py path/to/labels --save_csv stats.csv  # + CSV dump

The script expects each line to follow the YOLO convention::
    <class_id> <x_center> <y_center> <width> <height> [confidence]
Only the first token (integer class id) is required for counting.

Outputs
^^^^^^^
* total label files & how many are empty
* total number of objects
* objects per class (count & %)
* min/avg/max objects per image
* top-N images with zero labels (optional)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from collections import Counter, defaultdict

# -----------------------------------------------------------------------------
# Default 17-class mapping – edit if your dataset differs.
# -----------------------------------------------------------------------------
DEFAULT_CLASS_NAMES = [
    "경차/세단",
    "SUV/승합차",
    "트럭",
    "버스(소형, 대형)",
    "통학버스(소형,대형)",
    "경찰차",
    "구급차",
    "소방차",
    "견인차",
    "기타 특장차",
    "성인",
    "어린이",
    "오토바이",
    "자전거 / 기타 전동 이동체",
    "라바콘",
    "삼각대",
    "기타",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _scan_label_files(label_dir: Path) -> Tuple[Counter, List[int]]:
    """Return (class_counter, objs_per_img list)."""
    class_counter: Counter[int] = Counter()
    objs_per_img: List[int] = []

    txt_files = sorted(label_dir.glob("*.txt"))
    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except UnicodeDecodeError:
            with open(txt, "r", encoding="latin1") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]

        objs_per_img.append(len(lines))
        for ln in lines:
            try:
                cls_id = int(ln.split()[0])
                class_counter[cls_id] += 1
            except (IndexError, ValueError):
                # malformed – ignore but count as warning (class id -1)
                class_counter[-1] += 1

    return class_counter, objs_per_img


def _print_report(class_counter: Counter, objs_per_img: List[int], class_names: List[str]):
    total_imgs = len(objs_per_img)
    empty_imgs = sum(1 for n in objs_per_img if n == 0)
    total_objs = sum(class_counter.values())

    bar = "-" * 60
    print(bar)
    print(f"Total label files   : {total_imgs}")
    print(f"Files with 0 objects: {empty_imgs}  ({empty_imgs/total_imgs*100:.1f}%)")
    print(f"Total objects       : {total_objs}")
    print(f"Objects / image     : min {min(objs_per_img)}  avg {total_objs/total_imgs:.2f}  max {max(objs_per_img)}")
    print(bar)
    print(f"{'Class':<6} {'Name':<25} {'Count':>8}   %")
    print(bar)

    for cls_id in sorted(class_counter.keys()):
        if cls_id == -1:
            print(f"{cls_id:<6} {'<malformed>':<25} {class_counter[cls_id]:>8}")
            continue
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        cnt = class_counter[cls_id]
        pct = cnt / total_objs * 100 if total_objs else 0
        print(f"{cls_id:<6} {name[:24]:<25} {cnt:>8}  {pct:5.2f}%")
    print(bar)


def _save_csv(path: Path, class_counter: Counter, objs_per_img: List[int], class_names: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_id", "class_name", "count"])
        for cls_id in sorted(class_counter.keys()):
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
            writer.writerow([cls_id, name, class_counter[cls_id]])
        writer.writerow([])
        writer.writerow(["stat", "value"])
        writer.writerow(["total_images", len(objs_per_img)])
        writer.writerow(["empty_images", sum(1 for n in objs_per_img if n == 0)])
        writer.writerow(["total_objects", sum(class_counter.values())])
        writer.writerow(["min_objs_per_img", min(objs_per_img)])
        writer.writerow(["avg_objs_per_img", sum(class_counter.values()) / len(objs_per_img)])
        writer.writerow(["max_objs_per_img", max(objs_per_img)])


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple label-set inspector for YOLO txt files")
    ap.add_argument("label_dir", type=str, help="directory containing *.txt label files")
    ap.add_argument("--class_names", type=str, default=None,
                    help="Optional JSON or TXT file with class names (one per line) – overrides default 17-class list")
    ap.add_argument("--save_csv", type=str, default=None, help="Path to save summary CSV")
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    if not label_dir.is_dir():
        ap.error(f"label_dir {label_dir} is not a directory")

    # custom class names file? -------------------------------------------------
    class_names = DEFAULT_CLASS_NAMES
    if args.class_names:
        p = Path(args.class_names)
        if not p.exists():
            ap.error(f"class_names file {p} not found")
        if p.suffix.lower() == ".json":
            class_names = json.loads(p.read_text(encoding="utf-8"))
        else:
            class_names = [ln.strip() for ln in p.read_text("utf-8").splitlines() if ln.strip()]

    # scan & report -----------------------------------------------------------
    class_counter, objs_per_img = _scan_label_files(label_dir)
    _print_report(class_counter, objs_per_img, class_names)

    if args.save_csv:
        _save_csv(Path(args.save_csv), class_counter, objs_per_img, class_names)
        print(f"CSV saved → {args.save_csv}")


if __name__ == "__main__":
    main()
