from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import pandas as pd
import os
import random
from datetime import datetime

app = Flask(__name__)

# ── Load trained model and scaler ────────────────────────────────────────────
model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')

print("✅ Model loaded successfully!")

# ── Load dataset ─────────────────────────────────────────────────────────────
DATA_PATH = 'data/household_power_consumption.csv'

try:
    df = pd.read_csv(DATA_PATH, sep=';', na_values=['?'], low_memory=False)

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True
    )

    df.drop(['Date', 'Time'], axis=1, inplace=True)

    df.set_index('datetime', inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.ffill(inplace=True)

    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek

    # Daily averages
    df_daily = df.resample('D').mean().dropna()

    USE_DATASET = True

    print(f"✅ Dataset loaded: {len(df_daily)} daily records")

except Exception as e:
    USE_DATASET = False
    print(f"⚠️ Dataset not found — using simulation mode: {e}")

# ── Global Variables ─────────────────────────────────────────────────────────
current_index = [0]

TARIFF = 25.0

# Store recent predictions for smoothing
last_predictions = []

# ── Get live reading ─────────────────────────────────────────────────────────
def get_live_reading():

    if USE_DATASET:

        idx = current_index[0] % len(df_daily)

        row = df_daily.iloc[idx]

        # Move slowly through dataset
        current_index[0] += 1

        return {
            "voltage": round(float(row.get('Voltage', 230)), 2),

            "intensity": round(
                float(row.get('Global_intensity', 3.5)), 2
            ),

            "sub1": round(
                float(row.get('Sub_metering_1', 0)), 2
            ),

            "sub2": round(
                float(row.get('Sub_metering_2', 0)), 2
            ),

            "sub3": round(
                float(row.get('Sub_metering_3', 0)), 2
            ),

            "month": int(row.get('month', datetime.now().month)),

            "day_of_week": int(
                row.get('day_of_week', datetime.now().weekday())
            ),

            "timestamp": datetime.now().strftime("%H:%M:%S"),

            "source": "dataset"
        }

    else:
        # Stable simulation values
        return {
            "voltage": round(random.uniform(228, 232), 2),

            "intensity": round(random.uniform(2.5, 4.5), 2),

            "sub1": round(random.uniform(0.2, 1.0), 2),

            "sub2": round(random.uniform(0.1, 0.8), 2),

            "sub3": round(random.uniform(0.5, 1.5), 2),

            "month": datetime.now().month,

            "day_of_week": datetime.now().weekday(),

            "timestamp": datetime.now().strftime("%H:%M:%S"),

            "source": "simulation"
        }

# ── Home Route ───────────────────────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('index.html')

# ── Dataset Info Route ───────────────────────────────────────────────────────
@app.route('/api/dataset-info')
def dataset_info():

    if USE_DATASET:
        return jsonify({
            "loaded": True,
            "records": len(df_daily),

            "date_range":
                f"{df_daily.index[0].date()} "
                f"to {df_daily.index[-1].date()}",

            "current_row": current_index[0] % len(df_daily),

            "source": DATA_PATH
        })

    return jsonify({
        "loaded": False,
        "source": "simulation"
    })

# ── Live Prediction Route ────────────────────────────────────────────────────
@app.route('/api/live')
def live_data():

    global last_predictions

    reading = get_live_reading()

    # Features for prediction
    features = np.array([[
        reading['voltage'],
        reading['intensity'],
        reading['sub1'],
        reading['sub2'],
        reading['sub3'],
        reading['month'],
        reading['day_of_week']
    ]])

    # Scale features
    scaled = scaler.transform(features)

    # Predict power
    pred_power = abs(float(model.predict(scaled)[0]))

    # ── Smooth predictions ──────────────────────────────────────────────────
    last_predictions.append(pred_power)

    # Keep only last 10 readings
    if len(last_predictions) > 10:
        last_predictions.pop(0)

    # Average predictions
    smooth_power = sum(last_predictions) / len(last_predictions)

    # ── Energy Calculations ─────────────────────────────────────────────────
    pred_kwh_day = round(smooth_power * 24, 2)

    pred_kwh_mon = round(pred_kwh_day * 30, 2)

    est_bill = round(pred_kwh_mon * TARIFF, 2)

    return jsonify({

        "voltage": reading['voltage'],

        "current": reading['intensity'],

        "power_kw": round(smooth_power, 3),

        "daily_kwh": pred_kwh_day,

        "monthly_kwh": pred_kwh_mon,

        "est_bill_lkr": est_bill,

        "timestamp": reading['timestamp'],

        "source": reading['source']
    })

# ── Manual Prediction Route ──────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.json

    features = np.array([[
        data['voltage'],
        data['intensity'],
        data['sub1'],
        data['sub2'],
        data['sub3'],
        data['month'],
        data['day_of_week']
    ]])

    scaled = scaler.transform(features)

    pred_power = abs(float(model.predict(scaled)[0]))

    pred_kwh_day = round(pred_power * 24, 2)

    pred_kwh_mon = round(pred_kwh_day * 30, 2)

    est_bill = round(pred_kwh_mon * TARIFF, 2)

    return jsonify({

        "predicted_power_kw": round(pred_power, 3),

        "predicted_kwh_day": pred_kwh_day,

        "predicted_kwh_month": pred_kwh_mon,

        "estimated_bill_lkr": est_bill
    })

# ── Run App ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)