from xgboost import XGBRFClassifier
from src.logger import logging
from src.utils import save_object

def train_hair_model(X_train, y_train):
    model = XGBRFClassifier()

    logging.info("Started training for hair")
    model.fit(X_train, y_train)
    logging.info("Completed training for hair")

    save_object(file_path="artifacts/hair_model.pkl", obj=model)

