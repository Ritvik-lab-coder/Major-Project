import os
from src.components.preprocessing.hair_preprocessor import preprocess_hair
from src.components.preprocessing.nail_preprocessor import preprocess_nail
from src.components.preprocessing.teeth_preprocessor import preprocess_teeth
from src.exception import CustomException
from src.logger import logging

def validate_path(path: str) -> None:
    """Ensure the given path exists."""
    if not os.path.exists(path):
        raise CustomException(f"Path does not exist: {path}")

def ingest_data(category: str, input_base: str = "data", output_base: str = "data_preprocessed") -> None:
    """
    Main data ingestion function. It dispatches preprocessing based on the category.
    
    Args:
        category: one of ["hair", "nail", "teeth"]
        input_base: base directory containing raw data
        output_base: base directory to store preprocessed data
    """
    input_path = os.path.join(input_base, category)
    output_path = os.path.join(output_base, category)

    print(f"\n📥 Starting data ingestion for: {category}")
    validate_path(input_path)

    if category == "hair":
        logging.info("Started hair preprocessing")
        preprocess_hair(input_path, output_path)
        logging.info("Completed hair preprocessing")
    elif category == "nail":
        logging.info("Started nail preprocessing")
        preprocess_nail(input_path, output_path)
        logging.info("Completed nail preprocessing")
    elif category == "teeth":
        logging.info("Started teeth preprocessing")
        preprocess_teeth(input_path, output_path)
        logging.info("Completed teeth preprocessing")
    else:
        raise CustomException(f"Preprocessing for category '{category}' is not implemented yet.")

    print(f"📁 Preprocessed data stored at: {output_path}")

if __name__ == "__main__":
    # Calling ingest_data for each category
    categories = ["hair", "nail", "teeth"]
    for category in categories:
        ingest_data(category=category)
