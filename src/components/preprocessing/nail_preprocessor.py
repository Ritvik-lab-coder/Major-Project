import os
import cv2
import numpy as np
from tqdm import tqdm

TARGET_SIZE = (224, 224)

def apply_clahe(img):
    """
    Enhances local contrast using CLAHE on the L-channel in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_bilateral_filter(img):
    """
    Applies bilateral filtering to reduce noise while preserving edges.
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def preprocess_image(img):
    """
    Applies CLAHE, bilateral filtering, and normalization.
    """
    img = cv2.resize(img, TARGET_SIZE)
    img = apply_clahe(img)
    img = apply_bilateral_filter(img)
    img = img.astype(np.float32) / 255.0
    return (img * 255).astype(np.uint8)

def preprocess_nail(input_root: str, output_root: str):
    """
    Processes all disease folders directly under input_root.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for disease_folder in os.listdir(input_root):
        input_folder = os.path.join(input_root, disease_folder)
        output_folder = os.path.join(output_root, disease_folder)

        if not os.path.isdir(input_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        for image_name in tqdm(os.listdir(input_folder), desc=f"Processing {disease_folder}"):
            input_image_path = os.path.join(input_folder, image_name)
            output_image_path = os.path.join(output_folder, image_name)

            try:
                img = cv2.imread(input_image_path)
                if img is not None:
                    processed = preprocess_image(img)
                    cv2.imwrite(output_image_path, processed)
            except Exception as e:
                print(f"❌ Error processing {input_image_path}: {e}")
