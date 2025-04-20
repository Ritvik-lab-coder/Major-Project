# Instructions to run

1. Download the data from drive, unzip and place in the root directory
2. Install the required libraries by running `pip install -r requirements.txt`
3. Run the script `python -m src.components.ingest` from the root to start data ingestion
4. Run the following sql script to create database and tables
    - CREATE DATABASE nutriscan
    - CREATE TABLE nail (
        id SERIAL PRIMARY KEY,
        vector TEXT NOT NULL,
        label TEXT NOT NULL
    );
    - CREATE TABLE hair (
        id SERIAL PRIMARY KEY,
        vector TEXT NOT NULL,
        label TEXT NOT NULL
    );
    - CREATE TABLE teeth (
        id SERIAL PRIMARY KEY,
        vector TEXT NOT NULL,
        label TEXT NOT NULL
    );
5. Run the script `python -m src.components.transform` from the root to start data transformation