import joblib
import pandas as pd


MODEL_PATH = "ML/model.pkl"
DATA_PATH = "data/processed/processed_data.csv"


def test_model_can_be_loaded():

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_can_make_prediction():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("medv", axis=1)

    sample = X.iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1


def test_prediction_is_numeric():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("medv", axis=1)

    sample = X.iloc[[0]]

    prediction = model.predict(sample)

    assert isinstance(
        float(prediction[0]),
        float
    )