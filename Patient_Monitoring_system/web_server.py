from flask import Flask, render_template_string, jsonify
import time
import random
import threading
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from main import (
    calculate_qsofa, calculate_news2, bayesian_update,
    compute_utility, get_weights_for_patient, predictor
)

app = Flask(__name__)

# Global data storage for real-time updates
current_patients_data = []
chart_history = {
    'timestamps': [],
    'heart_rate': {},
    'resp_rate': {},
    'spo2': {},
    'temperature': {},
    'sbp': {},
    'qsofa': {},
    'news2': {},
    'ml_prob': {},
    'ml_risk': {}
}

# Initialize with CSV and random patients
csv_patients = []
random_patients = []

def initialize_patients():
    global csv_patients, random_patients
    from data_loader import CSVPatient
    from random_generator import RandomPatient

    csv_patients = [
        CSVPatient(os.path.join(BASE_DIR, f"patient{i}.csv"), i)
        for i in range(1, 6)
    ]

    random_patients = [
        RandomPatient(6),
        RandomPatient(7),
        RandomPatient(8),
        RandomPatient(9),
        RandomPatient(10),
    ]

def get_current_patient_data():
    global current_patients_data, chart_history

    all_patients = []

    # Get data from CSV patients
    for patient in csv_patients:
        vital = patient.get_next_vital()
        if vital:
            qsofa = calculate_qsofa(vital)
            news2 = calculate_news2(vital)

            # Determine if patient is high-risk using threshold
            is_high_risk = (qsofa >= 2 or news2 >= 7)

            # Bayesian probability of being critical
            prob_critical = bayesian_update(is_high_risk)

            # Stability now derived from Bayesian posterior
            stability = 1.0 - prob_critical

            response_time = news2 / 20.0
            resource_efficiency = 1.0 - 0.5  # environment_value
            af = 0  # alert_fatigue.get(vital["patient_id"], 0)

            ml_label, ml_prob = predictor.predict(vital)
            blended_stability = 0.5 * stability + 0.5 * (1.0 - ml_prob)

            weights = get_weights_for_patient(
                patient_id=vital["patient_id"],
                icu_bed_availability=0.5,
                risk_label=ml_label,
                alert_fatigue_map={},
            )

            utility = compute_utility(
                vital["patient_id"],
                blended_stability,
                response_time,
                resource_efficiency,
                af,
                weights,
            )

            patient_data = {
                'id': vital['patient_id'],
                'heart_rate': vital['heart_rate'],
                'resp_rate': vital['resp_rate'],
                'spo2': vital['spo2'],
                'temperature': vital['temperature'],
                'sbp': vital['sbp'],
                'consciousness': vital['consciousness'],
                'qsofa': qsofa,
                'news2': news2,
                'ml_risk': ml_label,
                'ml_prob': round(ml_prob * 100, 1),
                'utility': utility
            }
            all_patients.append(patient_data)

    # Get data from random patients
    for patient in random_patients:
        vital = patient.generate_vital()
        qsofa = calculate_qsofa(vital)
        news2 = calculate_news2(vital)

        # Determine if patient is high-risk using threshold
        is_high_risk = (qsofa >= 2 or news2 >= 7)

        # Bayesian probability of being critical
        prob_critical = bayesian_update(is_high_risk)

        # Stability now derived from Bayesian posterior
        stability = 1.0 - prob_critical

        response_time = news2 / 20.0
        resource_efficiency = 1.0 - 0.5
        af = 0

        # ML Risk Prediction
        ml_label, ml_prob = predictor.predict(vital)
        blended_stability = 0.5 * stability + 0.5 * (1.0 - ml_prob)

        weights = get_weights_for_patient(
            patient_id=vital["patient_id"],
            icu_bed_availability=0.5,
            risk_label=ml_label,
            alert_fatigue_map={},
        )

        utility = compute_utility(
            vital["patient_id"],
            blended_stability,
            response_time,
            resource_efficiency,
            af,
            weights,
        )

        patient_data = {
            'id': vital['patient_id'],
            'heart_rate': vital['heart_rate'],
            'resp_rate': vital['resp_rate'],
            'spo2': vital['spo2'],
            'temperature': vital['temperature'],
            'sbp': vital['sbp'],
            'consciousness': vital['consciousness'],
            'qsofa': qsofa,
            'news2': news2,
            'ml_risk': ml_label,
            'ml_prob': round(ml_prob * 100, 1),
            'utility': utility
        }
        all_patients.append(patient_data)

    # Update chart history
    current_time = len(chart_history['timestamps'])
    chart_history['timestamps'].append(current_time)

    # Keep only last 20 points
    if len(chart_history['timestamps']) > 20:
        chart_history['timestamps'] = chart_history['timestamps'][-20:]

    for patient in all_patients:
        pid = str(patient['id'])
        if pid not in chart_history['heart_rate']:
            chart_history['heart_rate'][pid] = []
            chart_history['resp_rate'][pid] = []
            chart_history['spo2'][pid] = []
            chart_history['temperature'][pid] = []
            chart_history['sbp'][pid] = []
            chart_history['qsofa'][pid] = []
            chart_history['news2'][pid] = []
            chart_history['ml_prob'][pid] = []
            chart_history['ml_risk'][pid] = []

        chart_history['heart_rate'][pid].append(patient['heart_rate'])
        chart_history['resp_rate'][pid].append(patient['resp_rate'])
        chart_history['spo2'][pid].append(patient['spo2'])
        chart_history['temperature'][pid].append(patient['temperature'])
        chart_history['sbp'][pid].append(patient['sbp'])
        chart_history['qsofa'][pid].append(patient['qsofa'])
        chart_history['news2'][pid].append(patient['news2'])
        chart_history['ml_prob'][pid].append(patient['ml_prob'])
        chart_history['ml_risk'][pid].append(patient['ml_risk'])

        # Keep only last 20 points
        for key in chart_history:
            if key != 'timestamps' and pid in chart_history[key]:
                chart_history[key][pid] = chart_history[key][pid][-20:]

    current_patients_data[:] = all_patients
    return all_patients

@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/api/patients')
def get_patients():
    return jsonify(get_current_patient_data())

@app.route('/api/charts')
def get_charts():
    return jsonify(chart_history)

@app.route("/patient_graph/<int:patient_id>")
def patient_graph(patient_id):
    pid = str(patient_id)

    if pid not in chart_history.get('heart_rate', {}):
        return jsonify({
            "timestamps": [],
            "heart_rate": [],
            "spo2": [],
            "temperature": [],
            "resp_rate": [],
            "sbp": []
        })

    return jsonify({
        "timestamps": chart_history["timestamps"],
        "heart_rate": chart_history["heart_rate"].get(pid, []),
        "spo2": chart_history["spo2"].get(pid, []),
        "temperature": chart_history["temperature"].get(pid, []),
        "resp_rate": chart_history.get("resp_rate", {}).get(pid, []),
        "sbp": chart_history.get("sbp", {}).get(pid, [])
    })

def update_data_loop():
    while True:
        get_current_patient_data()
        time.sleep(15)  # Update every 15 seconds

if __name__ == '__main__':
    initialize_patients()
    # Start background thread for data updates
    update_thread = threading.Thread(target=update_data_loop, daemon=True)
    update_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000)

    