import pandas as pd


DATA_PATH = "data/processed/processed_data.csv"


def test_processed_data_exists():

    df = pd.read_csv(DATA_PATH)

    assert not df.empty


def test_required_columns_exist():

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
        "medv"
    ]

    for column in required_columns:

        assert column in df.columns


def test_no_missing_values():

    df = pd.read_csv(DATA_PATH)

    assert df.isnull().sum().sum() == 0