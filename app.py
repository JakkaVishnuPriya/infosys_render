
from flask import Flask, request, render_template, redirect, url_for, flash, session
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import uuid
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'

# Load the pre-trained model, scaler, and feature names
MODEL_PATH = 'earthquake_rf_model.pkl'
SCALER_PATH = 'earthquake_scaler.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
    print(f"Successfully loaded scaler from {SCALER_PATH}")
    print(f"Successfully loaded feature names from {FEATURE_NAMES_PATH}")
except FileNotFoundError:
    print(f"Error: One or more necessary files ({MODEL_PATH}, {SCALER_PATH}, {FEATURE_NAMES_PATH}) not found.")
    print("Please ensure the model training script was run successfully and these files exist in the same directory as app.py.")
    exit()

# Define alert level mappings
alert_to_class = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}

class_to_alert = {v: k for k, v in alert_to_class.items()}

# Define alert colors for UI
ALERT_COLORS = {
    "green": "#10b981",  # Corresponds to --green-alert-bg
    "yellow": "#f59e0b", # Corresponds to --yellow-alert-bg
    "orange": "#f97316", # Corresponds to --orange-alert-bg
    "red": "#ef4444"     # Corresponds to --red-alert-bg
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/free_prediction', methods=['GET'])
def free_prediction():
    return render_template('free_prediction.html')

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    prediction_result = None
    probability_plot = None
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_id = str(uuid.uuid4())[:8] # Generate a short unique ID

    try:
        # Get input values from the form
        magnitude = float(request.form['magnitude'])
        depth = float(request.form['depth'])
        cdi = float(request.form['cdi'])
        mmi = float(request.form['mmi'])
        sig = float(request.form['sig'])

        # Create a DataFrame from the input
        input_data = pd.DataFrame([{
            'magnitude': magnitude,
            'depth': depth,
            'cdi': cdi,
            'mmi': mmi,
            'sig': sig
        }])

        # Ensure the input features are in the correct order
        input_data = input_data[feature_names]

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Make prediction
        predicted_class = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        predicted_alert = class_to_alert[predicted_class]
        predicted_color = ALERT_COLORS[predicted_alert]

        # Prepare probabilities for display
        prob_display = {class_to_alert[i].capitalize(): f"{p*100:.2f}%" for i, p in enumerate(probabilities)}

        prediction_result = {
            'alert': predicted_alert.upper(),
            'color': predicted_color,
            'probabilities': prob_display
        }

        # Generate probability plot
        fig, ax = plt.subplots(figsize=(8, 4))
        alerts = [class_to_alert[i].capitalize() for i in range(len(probabilities))]

        # Use hardcoded colors for the plot to maintain consistency visually, even if the UI background changes
        plot_colors = {
            'Green': '#10b981',
            'Yellow': '#f59e0b',
            'Orange': '#f97316',
            'Red': '#ef4444'
        }
        colors = [plot_colors[alert] for alert in alerts]

        ax.bar(alerts, probabilities, color=colors)
        # Use hardcoded colors for plot text and background for stability during debugging
        ax.set_title('Prediction Probabilities', fontsize=14, color='black') # Fixed color
        ax.set_xlabel('Alert Level', fontsize=12, color='black') # Fixed color
        ax.set_ylabel('Probability', fontsize=12, color='black') # Fixed color
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', colors='black') # Fixed color
        ax.tick_params(axis='y', colors='black') # Fixed color
        ax.set_facecolor('lightgray') # Fixed color
        fig.patch.set_facecolor('white') # Fixed color

        # Add percentage labels on top of bars
        for i, p in enumerate(probabilities):
            ax.text(i, p + 0.02, f'{p*100:.1f}%', ha='center', va='bottom', fontsize=10, color='black') # Fixed color

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        probability_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

    except ValueError:
        prediction_result = {'error': 'Invalid input. Please enter numeric values for all fields.'}
    except Exception as e:
        prediction_result = {'error': f'An error occurred during prediction: {e}'}

    return render_template('result.html', prediction_result=prediction_result, probability_plot=probability_plot, current_time=current_time, prediction_id=prediction_id, predicted_alert_class_name=predicted_alert.lower())

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Mock user check for demo
        if username == 'demo_user' and password == 'demo123':
            session['logged_in'] = True
            session['username'] = username  # Store username in session
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # For demonstration, we'll just acknowledge registration
        flash(f'User {username} registered successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in to access the dashboard.', 'danger')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
