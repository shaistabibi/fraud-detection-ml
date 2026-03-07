import joblib
import pandas as pd

# Load trained model from the models folder
MODEL_PATH = "models/fraud_model.pkl"
model = joblib.load(MODEL_PATH)

# Print model features
print("Model Features:")
print(model.feature_names_in_)

#transaction input
sample_transaction = {
    "id": 1,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}

# Convert input to DataFrame
input_df = pd.DataFrame([sample_transaction])

# Ensure columns match training features
input_df = input_df[model.feature_names_in_]

# Predict fraud
prediction = model.predict(input_df)
probability = model.predict_proba(input_df)

#result
print("\nPrediction Result")
print("----------------------")
print("Fraud Prediction:", prediction[0])
print("Fraud Probability:", probability[0][1])