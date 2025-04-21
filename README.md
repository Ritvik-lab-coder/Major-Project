# 🧠 NutriScan - Data Pipeline Instructions

Follow the steps below to run the full data pipeline for **NutriScan**, from data ingestion to feature extraction and database insertion.

---

## 📦 1. Setup

1. **Download and Extract Dataset**  
   - Download the dataset from the shared Google Drive link.  
   - Unzip the contents and place the extracted `data` folder in the **root directory** of the project.

2. **Install Dependencies**  
   Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 2. Data Ingestion

Run the data ingestion script to preprocess raw images:
```bash
python -m src.components.ingest
```

This will create a new folder named `data_preprocessed` containing cleaned images for **nail**, **hair**, and **teeth**.

---

## 📂 3. PostgreSQL Database Setup

Create a PostgreSQL database and required tables:

```sql
-- Create the database
CREATE DATABASE nutriscan;

-- Create table for Nail images
CREATE TABLE nail (
    id SERIAL PRIMARY KEY,
    vector TEXT NOT NULL,
    label TEXT NOT NULL
);

-- Create table for Hair images
CREATE TABLE hair (
    id SERIAL PRIMARY KEY,
    vector TEXT NOT NULL,
    label TEXT NOT NULL
);

-- Create table for Teeth images
CREATE TABLE teeth (
    id SERIAL PRIMARY KEY,
    vector TEXT NOT NULL,
    label TEXT NOT NULL
);
```

Ensure your PostgreSQL server is running and credentials are correctly set in the script before proceeding.

---

## 🔄 4. Data Transformation

Run the transformation script to convert images into HOG feature vectors and store them in the database:

```bash
python -m src.components.transform
```

This will:
- Convert all preprocessed images to feature vectors using MobileNetV2
- Store the vector and its associated label (disease) in the corresponding database table

---

## 🔄 5. Model Training

Run the model training script to train the models for nail, hair and teeth:

```bash
python -m src.components.train
```

This will:
- Train the model for nail, hair and teeth by fetching vectors from the database
- Convert the model to pkl file, and store in the artifacts table

---

