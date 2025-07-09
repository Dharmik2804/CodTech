# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature list
model, feature_cols = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # optional form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        input_df = pd.DataFrame([data], columns=feature_cols)
        input_df = input_df.astype(float)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
