from src.logger import logging
from sklearn.svm import SVC
from src.utils import save_object

def train_teeth_model(X_train, y_train):
    model = SVC()

    logging.info("Started training for teeth")
    model.fit(X_train, y_train)
    logging.info("Completed training for teeth")

    save_object(file_path="artifacts/teeth_model.pkl", obj=model)

