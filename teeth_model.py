import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    df = pd.read_sql("SELECT * FROM teeth", conn)
    conn.close()
    return df

# --- Preprocess vector column and split the dataset ---
def prepare_data(df):
    df['vector'] = df['vector'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    X = np.stack(df['vector'].values)
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Evaluate ML models ---
def evaluate_models(X_train, X_test, y_train, y_test):
    svc = SVC(probability=True)
    rf = RandomForestClassifier()
    voting = VotingClassifier(estimators=[('svc', svc), ('rf', rf)], voting='soft')

    models = {
        "SVC": svc,
        "Random Forest": rf,
        "Voting (SVC + RF)": voting
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

    # Use a sample only if you have more than 1000 rows
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)

    X_train, X_test, y_train, y_test = prepare_data(df)
    evaluate_models(X_train, X_test, y_train, y_test)