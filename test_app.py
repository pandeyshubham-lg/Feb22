from fastapi.testclient import TestClient
from app import app  # Make sure this path is correct

client = TestClient(app)

def test_predict_valid_input():
    # Provide valid numeric inputs as strings
    data = {
        "Pax": "100",
        "Temperature": "25",
        "AirDist": "500",
        "FlightTime": "2.5",
        "TripFuel": "200",
        "DepFuelPrice": "1.5",
        "DestFuelPrice": "1.6"
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert "Tankering Decision:" in response.text

def test_predict_invalid_input():
    # Provide an invalid value to trigger the ValueError exception
    data = {
        "Pax": "abc",  # non-numeric input
        "Temperature": "25",
        "AirDist": "500",
        "FlightTime": "2.5",
        "TripFuel": "200",
        "DepFuelPrice": "1.5",
        "DestFuelPrice": "1.6"
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert "Invalid input" in response.text
