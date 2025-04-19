import os
import cv2
import numpy as np
from tqdm import tqdm

# Constants
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")

def preprocess_image(image):
    """
    Preprocesses a single teeth image.
    Steps: grayscale → CLAHE → bilateral filter → normalize → convert to 3 channels → resize.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    filtered = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

    final_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(final_img, IMAGE_SIZE)

    return resized

def preprocess_teeth(input_base: str, output_base: str):
    """
    Recursively processes all valid image files under disease folders in input_base.
    """
    print("🔄 Starting preprocessing...\n")

    for folder in os.listdir(input_base):
        folder_path = os.path.join(input_base, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"📂 Processing: {folder}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(VALID_EXTENSIONS):
                    continue

                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_base)
                save_dir = os.path.join(output_base, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                try:
                    img = cv2.imread(input_path)
                    if img is not None:
                        processed_img = preprocess_image(img)
                        base_name = os.path.splitext(file)[0]
                        output_path = os.path.join(save_dir, base_name + ".png")
                        cv2.imwrite(output_path, processed_img)
                except Exception as e:
                    print(f"❌ Error processing {input_path}: {e}")

    print("\n✅ Teeth preprocessing complete.")
