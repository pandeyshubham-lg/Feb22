import joblib
import numpy as np
model = joblib.load('pred_tank.joblib')


input_features = np.array([[data.Pax, data.Temperature, data.AirDist, data.FlightTime,
                                data.TripFuel, data.DepFuelPrice, data.DestFuelPrice]])

model.predict(input_features)