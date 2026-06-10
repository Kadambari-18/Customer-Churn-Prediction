from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model files
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get form data as dictionary
    input_data = request.form.to_dict()

    # Convert numeric fields properly
    input_data['tenure'] = int(input_data['tenure'])
    input_data['MonthlyCharges'] = float(input_data['monthly'])
    input_data['TotalCharges'] = float(input_data['total'])
    input_data['SeniorCitizen'] = int(input_data['SeniorCitizen'])

    # Remove old form names (monthly & total)
    input_data.pop('monthly')
    input_data.pop('total')

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply One-Hot Encoding
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    # Prediction message
    if prediction == 1:
        result = "Customer is likely to CHURN ❌"
    else:
        result = "Customer is likely to STAY ✅"

    # Risk level classification
    if probability < 30:
        risk = "Low Risk 🟢"
    elif probability < 70:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    return render_template(
        'index.html',
        prediction_text=result,
        prob=round(probability, 2),
        risk_level=risk
    )


if __name__ == "__main__":
    app.run(debug=True)