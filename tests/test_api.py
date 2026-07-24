from fastapi.testclient import TestClient

from API.main import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():

    input_data = {

        "crim": 0.03,
        "zn": 0,
        "indus": 7.0,
        "chas": 0,
        "nox": 0.5,
        "rm": 6.0,
        "age": 60,
        "dis": 4.5,
        "rad": 4,
        "tax": 300,
        "ptratio": 18,
        "b": 390,
        "lstat": 10

    }

    response = client.post(
        "/predict",
        json=input_data
    )

    assert response.status_code == 200

    response_data = response.json()

    assert "predicted_medv" in response_data

    assert isinstance(
        response_data["predicted_medv"],
        float
    )


def test_invalid_input_is_rejected():

    invalid_data = {

        "crim": 0.03,
        "zn": 0

    }

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422