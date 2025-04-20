import os
import numpy as np
import psycopg2
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.exception import CustomException
from src.logger import logging

# --- DB CONFIG ---
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "nutriscan",
    "user": "postgres",
    "password": "postgres"
}

IMAGE_SIZE = (128, 128)
PCA_DIMENSIONS = 512  # Final vector size after PCA

def extract_hog_features(image_path):
    """
    Extracts raw HOG features from a grayscale image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise CustomException(f"Failed to read image: {image_path}")
    
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    features = hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

def apply_pca_to_features(all_features, n_components=PCA_DIMENSIONS):
    """
    Reduces the dimensionality of the feature set using PCA.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(all_features)
    return reduced_features

def insert_vector_to_db(table, vector_str, label, conn):
    """
    Inserts a vector and its label into the PostgreSQL table.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (vector, label) VALUES (%s, %s);",
            (vector_str, label)
        )
    conn.commit()

def process_category(category: str, base_dir: str = "data"):
    """
    Extracts HOG features, applies PCA, and inserts into DB for a category.
    """
    data_path = os.path.join(base_dir, category)
    validate_path(data_path)

    logging.info(f"🔄 Transforming and inserting data for: {category}")
    all_features = []
    labels = []
    image_paths = []

    for disease in os.listdir(data_path):
        disease_path = os.path.join(data_path, disease)
        if not os.path.isdir(disease_path):
            continue

        for img_name in os.listdir(disease_path):
            img_path = os.path.join(disease_path, img_name)
            try:
                features = extract_hog_features(img_path)
                all_features.append(features)
                labels.append(disease)
                image_paths.append(img_path)
            except Exception as e:
                logging.warning(f"❌ Error processing {img_path}: {e}")

    if not all_features:
        logging.error("No features extracted. Transformation aborted.")
        return

    reduced_features = apply_pca_to_features(np.array(all_features))

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        for i, vec in enumerate(reduced_features):
            vector_str = ",".join(map(str, vec))
            insert_vector_to_db(category, vector_str, labels[i], conn)
        conn.close()
        logging.info(f"✅ Successfully inserted reduced vectors for {category}")

    except Exception as e:
        raise CustomException(e)

def validate_path(path: str) -> None:
    if not os.path.exists(path):
        raise CustomException(f"Path does not exist: {path}")

if __name__ == "__main__":
    for cat in ["hair"]:
        process_category(cat)
