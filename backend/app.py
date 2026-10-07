from flask import Flask, request, jsonify
from flask_cors import CORS

# demand_model is imported so we can read the allowed category values from it
from prediction import make_prediction, demand_model


app = Flask(__name__)

# Allow frontend to communicate with backend
CORS(app)


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Smart Canteen ML API is running!"
    })


@app.route("/options", methods=["GET"])
def options():

    try:

        # The model's preprocessor remembers every category it saw in training
        preprocessor = demand_model.named_steps["preprocessor"]

        # Names of the categorical columns, e.g. Business_Type, Weather
        categorical_columns = preprocessor.transformers_[0][2]

        # The encoder that holds the allowed values for each column
        encoder = preprocessor.named_transformers_["categorical"]

        # Build {"Weather": ["Cold", "Hot", ...], ...}
        result = {
            column: categories.tolist()
            for column, categories in zip(
                categorical_columns,
                encoder.categories_
            )
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get data sent by frontend
        user_data = request.get_json()

        # Make prediction
        result = make_prediction(user_data)

        # Return prediction
        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )