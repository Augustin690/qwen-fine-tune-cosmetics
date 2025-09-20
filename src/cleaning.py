"""Script to clean the crawled csv_data: Deduplication of images, Removal of low-quality or invalid images,
Verification and transformation of annotation formats (if necessary), Deliver a well-structured and normalized dataset directory."""

import os
import hashlib
import shutil
import sys

from tqdm import tqdm
from datetime import datetime
from tkinter import BooleanVar

from PIL import Image, ImageStat
import pandas as pd
import cv2
import numpy as np
import json

def get_image_hash(file_path) -> str|None:
    """Perceptual hashing for near-duplicate detection"""
    try:
        img = Image.open(file_path)
        hash_size = 8
        img = img.convert("L").resize((hash_size+1, hash_size))
        pixels = list(img.getdata())
        diff = [pixels[i*hash_size + j] > pixels[i*hash_size + j + 1]
                for i in range(hash_size) for j in range(hash_size)]
        return hashlib.md5(str(diff).encode()).hexdigest()
    except:
        return None

def remove_duplicates(raw_dir:str, clean_dir:str)-> list:
    hashes = {}
    duplicates = []

    for root, _, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = get_image_hash(file_path)

            if not file_hash:
                continue

            if file_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[file_hash] = file_path
                shutil.copy(file_path, os.path.join(clean_dir, file))

    return duplicates

def quality_checks(image_path:str, var_threshold:int, min_size:int) ->tuple[bool,str]:
    """Check for blur, corruption, and minimum size"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Corrupted file"

        # Blur detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < var_threshold:
            return False, f"Blurry image (variance: {laplacian_var:.2f})"

        # Minimum size check
        if img.shape[0] < min_size or img.shape[1] < min_size:
            return False, f"Small image ({img.shape[0]}x{img.shape[1]})"

        return True, "OK"
    except Exception as e:
        return False, str(e)

def validate_annotations(ann_path:str, clean_image_dir:str) -> pd.DataFrame:
    # drop potential duplicated image_ids
    df = pd.read_csv(ann_path).drop_duplicates(subset=["image_id"], keep="first")
    valid_images = set(os.listdir(os.path.join(clean_image_dir, "images")))

    # Step 3: Extract image_id from file names and match with annotations
    def extract_image_id(file_name):
        # Extract image_id from {product_name}_{image_id}.png
        return "_".join(file_name.split("_")[-2:]).replace(".jpg", "")

    # Create a set of valid image_ids from the file names
    valid_image_ids = set(extract_image_id(file_name) for file_name in valid_images)

    # Step 4: Remove entries for missing images
    clean_df = df[df["image_id"].isin(valid_image_ids)]

    # Convert to standardized format (COCO example)
    coco_ann = {
        "images": [],
        "annotations": []
    }

    for idx, row in clean_df.iterrows():
        coco_ann["images"].append({
            "id": idx,
            "file_name": f"{row['image_id']}.jpg",
            "product_url": row["product_url"]
        })

    clean_df.to_csv(os.path.join(CLEAN_DIR, "annotations", "clean_annotations.csv"), index=False)
    with open(os.path.join(CLEAN_DIR, "annotations", "coco_format.json"), "w") as f:
        json.dump(coco_ann, f)

    return clean_df

def clean_dataset(laplacian_threshold:int, size_threshold:int) -> pd.DataFrame:
    print("Starting deduplication...")
    # Step 1: Deduplication
    duplicates = remove_duplicates(RAW_DIR, os.path.join(CLEAN_DIR, "images"))
    print(f"Removed {len(duplicates)} duplicates.")

    # Step 2: Quality Filtering
    print("Starting quality filtering...")
    quality_issues = []
    clean_images_dir = os.path.join(CLEAN_DIR, "images")
    image_files = os.listdir(clean_images_dir)

    for img_file in tqdm(image_files, desc="Checking image quality", unit="image"):
        img_path = os.path.join(CLEAN_DIR, "images", img_file)
        valid, reason = quality_checks(img_path, laplacian_threshold, size_threshold)
        if not valid:
            quality_issues.append((img_file, reason))
            os.remove(img_path)
    print(f"Removed {len(quality_issues)} images due to quality issues.")

    # Step 3: Annotation Validation
    print("Starting annotation validation...")
    validate_annotations(ANN_PATH, CLEAN_DIR)
    print("Annotations validated and cleaned.")

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""Cleaning Report (Generated at: {timestamp}):
    - Parameters:
        - Laplacian variance threshold: {laplacian_threshold}
        - Minimum size threshold: {size_threshold} pixels
    - Initial images: {len(os.listdir(RAW_DIR))}
    - Duplicates removed: {len(duplicates)}
    - Quality issues removed: {len(quality_issues)}
    - Final clean images: {len(os.listdir(os.path.join(CLEAN_DIR, "images")))}\n
    """

    with open(LOG_PATH, "a") as f:
        f.write(report)
        f.write("\nQuality Issues:\n")
        for img, reason in quality_issues:
            f.write(f"{img}: {reason}\n")
        f.write("\nDuplicates:\n")
        for duplicate in duplicates:
            f.write(f"{duplicate}\n")


    print(report)



if __name__ == "__main__":

    # Define paths
    RAW_DIR = "../product_images"
    CLEAN_DIR = "../clean_dataset"
    LOG_PATH = "../log/cleaning_report.log"
    ANN_PATH = "../csv_data/images_id_full.csv"

    # Create clean directory structure
    os.makedirs(os.path.join(CLEAN_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(CLEAN_DIR, "annotations"), exist_ok=True)

    laplacian_threshold = 60
    size_threshold = 300

    # lower blur threshold
    clean_dataset(laplacian_threshold, size_threshold)