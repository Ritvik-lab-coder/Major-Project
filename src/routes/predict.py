import os
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

from src.components.preprocessing import hair_preprocessor, nail_preprocessor, teeth_preprocessor

predict_bp = Blueprint("predict", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATHS = {
    "hair": os.path.join(BASE_DIR, "artifacts", "hair_model.pkl"),
    "nail": os.path.join(BASE_DIR, "artifacts", "nail_model.pkl"),
    "teeth": os.path.join(BASE_DIR, "artifacts", "teeth_model.pkl")
}

IMG_SIZE = (224, 224)

# Label mappings based on folder order
LABEL_MAPS = {
    "hair": [
        "Alopecia Areata",
        "Contact Dermatitis",
        "Folliculitis",
        "Head Lice",
        "Lichen Planus",
        "Male Pattern Baldness",
        "Psoriasis",
        "Seborrheic Dermatitis",
        "Telogen Effluvium",
        "Tinea Capitis"
    ],
    "nail": [
        "Acral Lentiginous Melanoma",
        "Blue Finger",
        "Clubbing",
        "Healthy",
        "Onychogryphosis",
        "Onychomycosis",
        "Pitting"
    ],
    "teeth": [
        "Calculus",
        "Gingivitis",
        "Hypodontia",
        "Mouth Ulcer",
        "Tooth Discoloration"
    ]
}

# Load MobileNetV2 base model for feature extraction
_base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=_base_model.input, outputs=_base_model.output)

def extract_deep_features(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb, axis=0)
    x = preprocess_input(x)
    features = feature_extractor.predict(x, verbose=0)
    return features.flatten()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files.get("image")
        image_type = request.form.get("type")

        if not img_file or image_type not in ["hair", "nail", "teeth"]:
            return jsonify({"error": "Invalid input"}), 400

        in_bytes = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(in_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Preprocess based on type
        if image_type == "hair":
            preprocessed_img = hair_preprocessor.preprocess_image(image)
        elif image_type == "nail":
            preprocessed_img = nail_preprocessor.preprocess_image(image)
        else:  # teeth
            preprocessed_img = teeth_preprocessor.preprocess_image(image)

        # Feature extraction
        features = extract_deep_features(preprocessed_img).reshape(1, -1)

        # Load model and predict
        model = joblib.load(MODEL_PATHS[image_type])
        prediction_index = model.predict(features)[0]

        # Convert prediction to readable label
        label = LABEL_MAPS[image_type][int(prediction_index)]

        return jsonify({
            "prediction_index": int(prediction_index),
            "prediction_label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500