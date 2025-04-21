from src.logger import logging
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.utils import save_object

def train_nail_model(X_train, y_train):
    model = VotingClassifier(
        estimators=[
            ('svm', SVC(probability=True)),
            ('knn', KNeighborsClassifier()),
        ],
        voting='soft'
    )

    logging.info("Started training for nail")
    model.fit(X_train, y_train)
    logging.info("Completed training for nail")

    save_object(file_path="artifacts/nail_model.pkl", obj=model)

