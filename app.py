import numpy as np
import joblib
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Load the trained model and scaler
model = joblib.load('tankering_model.pkl')
scaler = joblib.load('scaler.pkl')

# Template rendering
templates = Jinja2Templates(directory="templates")


# Pydantic model for API input validation
class PredictRequest(BaseModel):
    Pax: float
    Temperature: float
    AirDist: float
    FlightTime: float
    TripFuel: float
    DepFuelPrice: float
    DestFuelPrice: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(
        request: Request,
        Pax: float = Form(...),
        Temperature: float = Form(...),
        AirDist: float = Form(...),
        FlightTime: float = Form(...),
        TripFuel: float = Form(...),
        DepFuelPrice: float = Form(...),
        DestFuelPrice: float = Form(...)
):
    # Prepare the input features for the model
    input_features = np.array([[Pax, Temperature, AirDist, FlightTime, TripFuel, DepFuelPrice, DestFuelPrice]])

    # Scale the input features
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Interpret the result
    result = "Yes" if prediction[0] == 1 else "No"

    return templates.TemplateResponse(
        "home.html", {"request": request, "prediction_text": f"Tankering Decision: {result}"}
    )


@app.post("/predict_api")
async def predict_api(data: PredictRequest):
    # Prepare input features
    input_features = np.array([[data.Pax, data.Temperature, data.AirDist, data.FlightTime,
                                data.TripFuel, data.DepFuelPrice, data.DestFuelPrice]])

    # Scale input features
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)

    return {"Tankering Decision": "Yes" if prediction[0] == 1 else "No"}
