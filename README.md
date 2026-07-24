# OpenHousing - House Price Prediction Platform

## Overview

OpenHousing is a machine learning-based house price prediction platform developed for the Housing Observatory.

The project uses housing and socio-economic indicators to predict median house values using the Boston Housing dataset.

### Technologies

- Python
- Pandas
- Scikit-learn
- SQLite
- SQLAlchemy
- FastAPI
- Streamlit
- Pytest
- Docker
- Docker Compose
- GitHub Actions

---

## Architecture

```text
BostonHousing.csv
        |
        v
   ETL Pipeline
        |
        v
 Data Validation
        |
        v
 SQLite Database
        |
        v
  ML Model Training
        |
        v
    model.pkl
        |
        v
    FastAPI API
        |
        v
Streamlit Dashboard
        |
        v
 House Price Prediction