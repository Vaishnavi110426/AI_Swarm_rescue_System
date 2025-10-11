# data/prepare_dataset.py
"""
Helper to structure a custom YOLO dataset.
Input: folder of images and annotations in Pascal VOC or COCO or YOLO txt.
This script will create train/val splits and output the folder structure
expected by ultralytics YOLO (images/train, images/val, labels/train, labels/val).
"""

import os
import shutil
import random

def prepare_yolo_dataset(src_images_dir, src_labels_dir, out_dir, val_split=0.2, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    images_train = os.path.join(out_dir, "images", "train")
    images_val = os.path.join(out_dir, "images", "val")
    labels_train = os.path.join(out_dir, "labels", "train")
    labels_val = os.path.join(out_dir, "labels", "val")
    for p in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(p, exist_ok=True)

    images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.seed(seed)
    random.shuffle(images)
    split = int(len(images) * (1 - val_split))
    train_files = images[:split]
    val_files = images[split:]

    for fname in train_files:
        shutil.copy(os.path.join(src_images_dir, fname), os.path.join(images_train, fname))
        lbl = os.path.splitext(fname)[0] + ".txt"
        if os.path.exists(os.path.join(src_labels_dir, lbl)):
            shutil.copy(os.path.join(src_labels_dir, lbl), os.path.join(labels_train, lbl))

    for fname in val_files:
        shutil.copy(os.path.join(src_images_dir, fname), os.path.join(images_val, fname))
        lbl = os.path.splitext(fname)[0] + ".txt"
        if os.path.exists(os.path.join(src_labels_dir, lbl)):
            shutil.copy(os.path.join(src_labels_dir, lbl), os.path.join(labels_val, lbl))

    print("Dataset prepared at:", out_dir)
