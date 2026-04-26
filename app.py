from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model + encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Extract OLD features
        Age = float(data.get('Age', 0))
        RBC_Count = float(data.get('RBC_Count', 0))
        Hemoglobin = float(data.get('Hemoglobin', 0))
        WBC_Count = float(data.get('WBC_Count', 0))
        Lymphocytes = float(data.get('Lymphocytes', 0))

        # Extract NEW features (autoimmune-related)
        PLT = float(data.get('PLT_Count', 0))
        ESR = float(data.get('ESR', 0))
        CRP = float(data.get('CRP', 0))
        ANA = float(data.get('ANA', 0))

        Family = int(data.get('Family_History', 0))
        Gender = 1 if data.get('Gender') == "Male" else 0
        Duration = float(data.get('Sickness_Duration_Months', 0))

        # Final feature vector (ORDER MUST MATCH TRAINING DATA)
        input_data = [
            Age,
            RBC_Count,
            Hemoglobin,
            WBC_Count,
            Lymphocytes,
            PLT,
            ESR,
            CRP,
            ANA,
            Family,
            Gender,
            Duration
        ]

        # Convert to numpy array
        input_array = np.array([input_data])

        # Prediction
        probs = model.predict_proba(input_array)[0]
        prediction = np.argmax(probs)
        risk = round(max(probs) * 100, 2)
        disease = label_encoder.inverse_transform([prediction])[0]

        return render_template(
            "index.html",
            result=disease,
            risk=risk
        )

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)