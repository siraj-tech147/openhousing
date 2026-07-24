import pandas as pd

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------

DATABASE_URL = "sqlite:///database/housing.db"

MODEL_PATH = "ML/model.pkl"


# -----------------------------
# LOAD DATA FROM SQLITE
# -----------------------------

print("Loading data from SQLite...")

engine = create_engine(
    DATABASE_URL
)

df = pd.read_sql(
    "SELECT * FROM housing_data",
    engine
)

print(
    f"Dataset shape: {df.shape}"
)


# -----------------------------
# FEATURES AND TARGET
# -----------------------------

X = df.drop(
    "medv",
    axis=1
)

y = df["medv"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


print(
    f"Training data: {X_train.shape}"
)

print(
    f"Testing data: {X_test.shape}"
)


# -----------------------------
# LINEAR REGRESSION
# -----------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print("\nLinear Regression Results")

print(
    f"MAE: {linear_mae:.2f}"
)

print(
    f"RMSE: {linear_rmse:.2f}"
)

print(
    f"R2 Score: {linear_r2:.2f}"
)


# -----------------------------
# RANDOM FOREST
# -----------------------------

random_forest_model = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)


random_forest_model.fit(

    X_train,

    y_train
)


rf_predictions = random_forest_model.predict(

    X_test
)


rf_mae = mean_absolute_error(

    y_test,

    rf_predictions
)


rf_rmse = mean_squared_error(

    y_test,

    rf_predictions

) ** 0.5


rf_r2 = r2_score(

    y_test,

    rf_predictions
)


print("\nRandom Forest Results")

print(

    f"MAE: {rf_mae:.2f}"

)

print(

    f"RMSE: {rf_rmse:.2f}"

)

print(

    f"R2 Score: {rf_r2:.2f}"

)


# -----------------------------
# SAVE BEST MODEL
# -----------------------------

print("\nBest Model: Random Forest")


Path("ML").mkdir(

    exist_ok=True

)


joblib.dump(

    random_forest_model,

    MODEL_PATH

)


print(

    f"Model saved to: {MODEL_PATH}"

)