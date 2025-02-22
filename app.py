from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = joblib.load("pred_tank.joblib")

@app.post("/predict")
async def predict(
    request: Request,
    Pax: str = Form(...),
    Temperature: str = Form(...),
    AirDist: str = Form(...),
    FlightTime: str = Form(...),
    TripFuel: str = Form(...),
    DepFuelPrice: str = Form(...),
    DestFuelPrice: str = Form(...)
):
    try:
        # Convert inputs to float
        input_features = np.array([[
            float(Pax), float(Temperature), float(AirDist), float(FlightTime),
            float(TripFuel), float(DepFuelPrice), float(DestFuelPrice)
        ]])

        # Make prediction
        prediction = model.predict(input_features)
        result = "Yes" if prediction[0] == 1 else "No"

        # Check if the request is from an API call (JSON expected)
        if request.headers.get("accept") == "application/json":
            return JSONResponse(content={"tankering_decision": result})

        # Otherwise, return HTML (for web users)
        return templates.TemplateResponse(
            request, "home.html", {"request": request, "prediction_text": f"Tankering Decision: {result}"}
        )
    except ValueError:
        return JSONResponse(
            content={"error": "Invalid input. Please enter valid numbers."}, status_code=400
        )
