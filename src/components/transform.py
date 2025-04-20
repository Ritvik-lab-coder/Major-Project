import os
import numpy as np
import psycopg2
import cv2
from skimage.feature import hog
from tqdm import tqdm
from src.exception import CustomException
from src.logger import logging

# --- DB CONFIG (set your credentials here) ---
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "nutriscan",
    "user": "postgres",
    "password": "postgres"
}

IMAGE_SIZE = (128, 128)  # Resize image for consistent HOG extraction size
VECTOR_SIZE = 512  # Fixed vector size for HOG features

def extract_hog_features(image_path):
    """
    Extracts HOG features from an image, resizes it, and normalizes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise CustomException(f"Failed to read image: {image_path}")
    
    # Resize image for consistent feature extraction
    image_resized = cv2.resize(image, IMAGE_SIZE)
    
    # Convert to grayscale if working with single channel
    # image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    # Using image_gray for grayscale or image_resized for color images
    features, _ = hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    # Ensure the features are of fixed size (VECTOR_SIZE), truncate or pad
    features = features[:VECTOR_SIZE]  # Truncate if necessary
    if len(features) < VECTOR_SIZE:
        features = np.pad(features, (0, VECTOR_SIZE - len(features)), mode='constant', constant_values=0)  # Pad if less
    
    return ",".join(map(str, features))  # Convert the vector to a comma-separated string for storage


def insert_vector_to_db(table, vector_str, label, conn):
    """
    Inserts a vector and its label into the given PostgreSQL table.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (vector, label) VALUES (%s, %s);",
            (vector_str, label)
        )
    conn.commit()

def process_category(category: str, base_dir: str = "data"):
    """
    Processes a category (nail, hair, teeth), extracts HOG features,
    and stores them in the corresponding table.
    """
    data_path = os.path.join(base_dir, category)
    validate_path(data_path)

    logging.info(f"🔄 Transforming and inserting data for: {category}")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        for disease in os.listdir(data_path):
            disease_path = os.path.join(data_path, disease)
            if not os.path.isdir(disease_path):
                continue

            for img_name in tqdm(os.listdir(disease_path), desc=f"Processing {disease}"):
                img_path = os.path.join(disease_path, img_name)
                try:
                    vector_str = extract_hog_features(img_path)
                    insert_vector_to_db(category, vector_str, disease, conn)
                except Exception as e:
                    logging.warning(f"❌ Error processing {img_path}: {e}")
        conn.close()
        logging.info(f"✅ Finished transformation for: {category}")

    except Exception as e:
        raise CustomException(e)

def validate_path(path: str) -> None:
    if not os.path.exists(path):
        raise CustomException(f"Path does not exist: {path}")

if __name__ == "__main__":
    for cat in ["hair","teeth","nail"]:
        process_category(cat)