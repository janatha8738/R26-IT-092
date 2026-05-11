from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import random
from datetime import datetime

app = Flask(__name__)

# ── Load PP1 trained model (3 features, first year UCI) ──────────────────────
model  = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')
print("✅ PP1 Model loaded — MLP (3 features)")

TARIFF = 25.0

# ── Simulate live sensor reading ─────────────────────────────────────────────
def get_live_reading():
    return {
        "voltage":     round(random.uniform(220, 240), 2),
        "intensity":   round(random.uniform(1.0, 8.0),  2),
        "sub1":        round(random.uniform(0, 2),      2),
        "sub2":        round(random.uniform(0, 1.5),    2),
        "sub3":        round(random.uniform(0, 3),      2),
        "month":       datetime.now().month,
        "day_of_week": datetime.now().weekday(),
        "timestamp":   datetime.now().strftime("%H:%M:%S")
    }

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/live')
def live_data():
    reading = get_live_reading()

    # PP1 model — only 3 features
    features = np.array([[
        reading['intensity'],  # Global_intensity
        reading['sub1'],       # Sub_metering_1
        reading['sub3'],       # Sub_metering_3
    ]])

    scaled       = scaler.transform(features)
    pred_power   = abs(float(model.predict(scaled)[0]))
    pred_kwh_day = round(pred_power * 24, 2)
    pred_kwh_mon = round(pred_kwh_day * 30, 2)
    est_bill     = round(pred_kwh_mon * TARIFF, 2)

    return jsonify({
        "voltage":      reading['voltage'],
        "current":      reading['intensity'],
        "power_kw":     round(pred_power, 3),
        "daily_kwh":    pred_kwh_day,
        "monthly_kwh":  pred_kwh_mon,
        "est_bill_lkr": est_bill,
        "timestamp":    reading['timestamp']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    # PP1 model — only 3 features
    features = np.array([[
        data['intensity'],   # Global_intensity
        data['sub1'],        # Sub_metering_1
        data['sub3'],        # Sub_metering_3
    ]])

    scaled       = scaler.transform(features)
    pred_power   = abs(float(model.predict(scaled)[0]))
    pred_kwh_mon = round(pred_power * 24 * 30, 2)
    est_bill     = round(pred_kwh_mon * TARIFF, 2)

    return jsonify({
        "predicted_power_kw":  round(pred_power, 3),
        "predicted_kwh_month": pred_kwh_mon,
        "estimated_bill_lkr":  est_bill
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)