import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = (224, 224)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applies grayscale conversion, CLAHE, bilateral filtering, 
    normalization, resizing, and converts back to 3 channels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    filtered = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

    final_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(final_img, IMAGE_SIZE)

    return resized

def preprocess_hair(input_dir: str, output_dir: str) -> None:
    """
    Processes each disease folder inside `input_dir` and saves preprocessed images
    into the mirrored folder structure in `output_dir`.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path {input_dir} does not exist or is not a directory.")

    for disease_name in os.listdir(input_dir):
        disease_path = os.path.join(input_dir, disease_name)
        if not os.path.isdir(disease_path):
            continue  # Skip files

        output_disease_path = os.path.join(output_dir, disease_name)
        os.makedirs(output_disease_path, exist_ok=True)

        print(f"Processing disease: {disease_name}...")

        for img_name in tqdm(os.listdir(disease_path), desc=f"  {disease_name}"):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")):
                img_path = os.path.join(disease_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        processed_img = preprocess_image(img)
                        output_path = os.path.join(output_disease_path, os.path.splitext(img_name)[0] + ".png")
                        cv2.imwrite(output_path, processed_img)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\n✅ Hair disease image preprocessing complete!")
