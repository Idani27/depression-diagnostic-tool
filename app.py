from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature list (must match training order)
feature_names = [
    'Sleep', 'Appetite', 'Interest', 'Fatigue', 'Worthlessness',
    'Concentration', 'Agitation', 'Suicidal Ideation', 'Sleep Disturbance',
    'Aggression', 'Panic Attacks', 'Hopelessness', 'Restlessness', 'Low Energy'
]

# Mapping values from form to model-friendly format
mapping = {
    1: 0,  # Never
    6: 0,  # Not at all
    4: 1,  # Rarely
    5: 2,  # Sometimes
    3: 3,  # Often
    2: 4   # Always
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inputs = {}  # to hold selected form values

    if request.method == 'POST':
        try:
            inputs = request.form.to_dict()

            # Apply mapping to each value
            mapped_values = [mapping[int(inputs[f])] for f in feature_names]

            # Scale values and make prediction
            scaled_values = scaler.transform([mapped_values])
            pred = model.predict(scaled_values)[0]
            prediction = "Depressed" if pred == 1 else "Not Depressed"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        features=feature_names,
        inputs=inputs
    )

if __name__ == '__main__':
    app.run(debug=True)
