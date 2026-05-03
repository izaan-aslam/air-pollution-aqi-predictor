from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

scaler_path = "scaler.pkl"
if os.path.getsize(scaler_path) == 0:
    raise ValueError("scaler.pkl is empty or corrupted. Recreate it.")

scaler = pickle.load(open(scaler_path, "rb"))



CITIES_DATA = [
    {"city": "Reykjavik",   "country": "Iceland",       "aqi": 10},
    {"city": "Zurich",      "country": "Switzerland",   "aqi": 18},
    {"city": "Sydney",      "country": "Australia",     "aqi": 25},
    {"city": "Toronto",     "country": "Canada",        "aqi": 35},
    {"city": "Berlin",      "country": "Germany",       "aqi": 45},
    {"city": "London",      "country": "UK",            "aqi": 60},
    {"city": "Los Angeles", "country": "USA",           "aqi": 80},
    {"city": "São Paulo",   "country": "Brazil",        "aqi": 100},
    {"city": "Bangkok",     "country": "Thailand",      "aqi": 120},
    {"city": "Mexico City", "country": "Mexico",        "aqi": 140},
    {"city": "Istanbul",    "country": "Turkey",        "aqi": 160},
    {"city": "Chengdu",     "country": "China",         "aqi": 180},
    {"city": "Jakarta",     "country": "Indonesia",     "aqi": 220},
    {"city": "Mumbai",      "country": "India",         "aqi": 260},
    {"city": "Cairo",       "country": "Egypt",         "aqi": 300},
    {"city": "Beijing",     "country": "China",         "aqi": 340},
    {"city": "Karachi",     "country": "Pakistan",      "aqi": 380},
    {"city": "Dhaka",       "country": "Bangladesh",    "aqi": 420},
    {"city": "Lahore",      "country": "Pakistan",      "aqi": 460},
    {"city": "New Delhi",   "country": "India",         "aqi": 480},
]


RAW_MAX = {
    "co":    50,
    "ozone": 200,
    "no2":   100,
    "pm25":  500
}
AQI_MAX = 500


def predict_aqi(co, ozone, no2, pm25):
    co_n    = co    / RAW_MAX["co"]
    ozone_n = ozone / RAW_MAX["ozone"]
    no2_n   = no2   / RAW_MAX["no2"]
    pm25_n  = pm25  / RAW_MAX["pm25"]

    X        = np.array([[co_n, ozone_n, no2_n, pm25_n]])
    X_scaled = scaler.transform(X)
    result   = model.predict(X_scaled)[0]
    return result * AQI_MAX


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        def clean(value, min_v=0, max_v=500):
            value = float(value)
            return max(min_v, min(value, max_v))

        co    = clean(data.get("co",    0), 0, 50)
        ozone = clean(data.get("ozone", 0), 0, 200)
        no2   = clean(data.get("no2",   0), 0, 100)
        pm25  = clean(data.get("pm25",  0), 0, 500)

        result = predict_aqi(co, ozone, no2, pm25)

        if result <= 50:
            category = "Good"
            tip = "Air is clean and safe for outdoor activity."
        elif result <= 100:
            category = "Moderate"
            tip = "Acceptable air quality. Sensitive people should be careful."
        elif result <= 150:
            category = "Unhealthy for Sensitive Groups"
            tip = "Sensitive groups may experience health effects."
        elif result <= 200:
            category = "Unhealthy"
            tip = "Everyone may experience health effects."
        else:
            category = "Hazardous"
            tip = "Emergency conditions. Avoid outdoor activity."

        return jsonify({
            "aqi":        round(result, 2),
            "category":   category,
            "health_tip": tip
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scale", methods=["POST"])
def scale():
    try:
        data = request.get_json(force=True)

        co    = float(data.get("co",    0))
        ozone = float(data.get("ozone", 0))
        no2   = float(data.get("no2",   0))
        pm25  = float(data.get("pm25",  0))

        co_n    = co    / RAW_MAX["co"]
        ozone_n = ozone / RAW_MAX["ozone"]
        no2_n   = no2   / RAW_MAX["no2"]
        pm25_n  = pm25  / RAW_MAX["pm25"]

        X        = np.array([[co_n, ozone_n, no2_n, pm25_n]])
        X_scaled = scaler.transform(X)[0]

        return jsonify({
            "co":    X_scaled[0],
            "ozone": X_scaled[1],
            "no2":   X_scaled[2],
            "pm25":  X_scaled[3]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cities")
def cities():
    try:
        aqi_target = float(request.args.get("aqi", 100))

        # Sort all cities by how close their AQI is to the target
        sorted_cities = sorted(
            CITIES_DATA,
            key=lambda c: abs(c["aqi"] - aqi_target)
        )

        # Return top 5 closest
        return jsonify(sorted_cities[:5])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)