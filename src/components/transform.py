import os
import numpy as np
import psycopg2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
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

# --- CNN Feature Extractor Setup (MobileNetV2) ---
IMG_SIZE = (224, 224)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_deep_features(img_path):
    """
    Extracts deep features from image using MobileNetV2 (pretrained).
    """
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array, verbose=0)
        features = features.flatten()
        return ",".join(map(str, features))  # Store as comma-separated string

    except Exception as e:
        raise CustomException(f"Failed to extract features from {img_path}: {e}")

def insert_vector_to_db(table, vector_str, label, conn):
    """
    Inserts a vector and its label into the specified table.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (vector, label) VALUES (%s, %s);",
            (vector_str, label)
        )
    conn.commit()

def validate_path(path: str):
    if not os.path.exists(path):
        raise CustomException(f"Path does not exist: {path}")

def process_category(category: str, base_dir: str = "data"):
    """
    Process and store deep features of all images in the given category.
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
                    vector_str = extract_deep_features(img_path)
                    insert_vector_to_db(category, vector_str, disease, conn)
                except Exception as e:
                    logging.warning(f"❌ Error processing {img_path}: {e}")
        conn.close()
        logging.info(f"✅ Finished transformation for: {category}")

    except Exception as e:
        raise CustomException(e)

# --- Main Runner ---
if __name__ == "__main__":
    for cat in ["nail", "hair", "teeth"]:
        process_category(cat)
