import pandas as pd
from sqlalchemy import create_engine


engine = create_engine(
    "sqlite:///database/housing.db"
)


df = pd.read_sql(
    "SELECT * FROM housing_data",
    engine
)


print(df.head())

print("Database shape:", df.shape)