import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Database Config ---
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "nutriscan",
    "user": "postgres",
    "password": "Shriram@321"
}

# --- Fetch Data from PostgreSQL ---
def fetch_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM nail", conn)
    conn.close()
    return df

# --- Preprocess vector column and label encode classes ---
def prepare_data(df):
    df['vector'] = df['vector'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    X = np.stack(df['vector'].values)

    # Label encode the labels
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Evaluate ML models ---
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "SVC": SVC(),
        "XGBRF": XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "MLP": MLPClassifier(max_iter=500)
    }

    for name, model in models.items():
        print(f"\n🔍 Training: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ {name} Accuracy: {acc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_test, preds))

# --- Main Execution ---
if __name__ == "__main__":
    df = fetch_data()

    print(f"\n📦 Total rows in DB: {len(df)}")
    print("📈 Label distribution:\n", df['label'].value_counts())

    # Prepare the data and split it
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Evaluate models
    evaluate_models(X_train, X_test, y_train, y_test)