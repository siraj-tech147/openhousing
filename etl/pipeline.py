import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


# -----------------------------
# Configuration
# -----------------------------

DATABASE_URL = "sqlite:///database/housing.db"

engine = create_engine(DATABASE_URL)

RAW_DATA_PATH = "data/raw/BostonHousing.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"


# -----------------------------
# EXTRACT
# -----------------------------

def extract_data():

    print("Extracting data...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# -----------------------------
# TRANSFORM
# -----------------------------

def transform_data(df):

    print("Transforming data...")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    # Convert all columns to numeric values
    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows that became invalid
    df = df.dropna()

    return df


# -----------------------------
# VALIDATE
# -----------------------------

def validate_data(df):

    print("Validating data...")

    # Check missing values
    assert df.isnull().sum().sum() == 0, \
        "Dataset contains missing values"

    # Check target column
    assert "medv" in df.columns, \
        "Target column 'medv' is missing"

    # Check dataset is not empty
    assert len(df) > 0, \
        "Dataset is empty"

    print("Data validation successful!")


# -----------------------------
# LOAD
# -----------------------------

def load_data(df):

    print("Loading processed data...")

    # Create processed data folder
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    # Create database folder
    Path("database").mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed CSV
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"Processed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    # Load data into SQLite
    print("Loading data into SQLite...")

    df.to_sql(
        "housing_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    print(
        "Data loaded into SQLite successfully!"
    )


# -----------------------------
# MAIN ETL PIPELINE
# -----------------------------

def run_pipeline():

    print("Starting ETL pipeline...\n")

    # Extract
    df = extract_data()

    # Transform
    df = transform_data(df)

    # Validate
    validate_data(df)

    # Load
    load_data(df)

    print(
        "\nETL pipeline completed successfully!"
    )


if __name__ == "__main__":

    run_pipeline()