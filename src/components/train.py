import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.components.models.hair_model import train_hair_model
from src.components.models.nail_model import train_nail_model
from src.components.models.teeth_model import train_teeth_model

# --- Database Config ---
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "nutriscan",
    "user": "postgres",
    "password": "postgres"
}

# --- Fetch Data from PostgreSQL ---
def fetch_data(table):
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

# --- Preprocess vector column and label encode classes ---
def prepare_data(df):
    # Convert the 'vector' column from string representation of lists to numpy arrays
    df['vector'] = df['vector'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    
    X = np.stack(df['vector'].values)

    # Label encode the 'label' column
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    return X, y

if __name__ == "__main__":
    df_hair = fetch_data("hair")
    X_train, y_train = prepare_data(df_hair)
    train_hair_model(X_train=X_train, y_train=y_train)

    df_nail = fetch_data("nail")
    X_train, y_train = prepare_data(df_nail)
    train_nail_model(X_train=X_train, y_train=y_train)

    df_teeth = fetch_data("teeth")
    X_train, y_train = prepare_data(df_teeth)
    train_teeth_model(X_train=X_train, y_train=y_train)

