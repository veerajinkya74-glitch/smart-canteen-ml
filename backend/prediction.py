import os
import joblib
import pandas as pd


# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
WASTE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "waste_prediction_model.pkl"
)

DEMAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "demand_prediction_model.pkl"
)


# Load models
waste_model = joblib.load(WASTE_MODEL_PATH)
demand_model = joblib.load(DEMAND_MODEL_PATH)


def make_prediction(user_data):

    # Convert user input into DataFrame
    input_data = pd.DataFrame([user_data])

    # Demand prediction
    predicted_demand = demand_model.predict(input_data)[0]

    # Waste prediction
    predicted_waste = waste_model.predict(input_data)[0]

    # Prevent negative predictions
    predicted_demand = max(0, predicted_demand)
    predicted_waste = max(0, predicted_waste)

    # Determine waste risk
    if predicted_waste > 10:
        waste_risk = "HIGH"

    elif predicted_waste > 5:
        waste_risk = "MEDIUM"

    else:
        waste_risk = "LOW"

    # Recommended preparation
    recommended_quantity = round(predicted_demand)

    # Recommendation message
    if waste_risk == "HIGH":
        recommendation = (
            "High waste risk. Reduce preparation quantity "
            "and monitor demand carefully."
        )

    elif waste_risk == "MEDIUM":
        recommendation = (
            "Medium waste risk. Monitor preparation quantity "
            "and actual sales."
        )

    else:
        recommendation = (
            "Low waste risk. Current preparation level "
            "looks reasonable."
        )

    return {
        "predicted_demand": round(predicted_demand, 2),
        "predicted_waste": round(predicted_waste, 2),
        "recommended_quantity": recommended_quantity,
        "waste_risk": waste_risk,
        "recommendation": recommendation
    }