# Credit Card Fraud Detection System

## Overview

This project is an end-to-end machine learning system built to detect fraudulent credit card transactions. The goal is to demonstrate how a real-world fraud detection pipeline can be built using Python, from data processing and model training to exposing predictions through an API.

The system uses a Random Forest classifier trained on historical credit card transaction data. After training, the model is saved and served through a FastAPI endpoint so that new transactions can be evaluated in real time.

This project was built as a practical exercise in applying machine learning to a real business problem while keeping the architecture simple and production-oriented.

---

## Why This Project

Credit card fraud is a serious problem for banks and payment companies. Even a small percentage of fraudulent transactions can lead to large financial losses.

Manual monitoring of transactions is not scalable, which is why companies rely on automated machine learning systems to detect suspicious activity.

The goal of this project is to build a basic version of such a system that can:

* Analyze transaction features
* Predict whether a transaction is fraudulent
* Provide a confidence score for the prediction
* Serve predictions through an API that other applications could use

---

## Tech Stack

Python
Pandas
Scikit-learn
FastAPI
Joblib
Uvicorn

These tools were chosen because they are widely used in production machine learning systems.

---

## Project Structure

```
## Project Structure

```
fraud-detection-ml
│
├── data
│   └── fraud_data.csv
│
├── ml
│
├── models
│   ├── fraud_model.pkl
│   └── model.pkl
│
├── notebooks
│   └── fraud_analyz.ipynb
│
├── src
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

### Folder Explanation

**data/**
Contains the dataset used to train and test the fraud detection model.

**ml/**
Reserved for additional machine learning utilities or experimentation scripts if the project grows.

**models/**
Stores trained machine learning models. These `.pkl` files are generated after training and are loaded when making predictions.

**notebooks/**
Contains Jupyter notebooks used for data exploration, visualization, and early experimentation before implementing the final pipeline.

**src/**
Core machine learning scripts.

* `train.py` – handles data loading, preprocessing, model training, and evaluation
* `predict.py` – loads the trained model and performs predictions on new data

**app.py**
FastAPI application that exposes the trained model through an API endpoint for real-time fraud prediction.

**requirements.txt**
Lists all Python dependencies required to run the project.

**README.md**
Documentation explaining the project, setup instructions, and usage.


---
## Key Features

• Machine learning model for fraud detection
• Handles imbalanced datasets using class weighting
• Uses Random Forest Classifier for reliable predictions
• Model serialization using Joblib
• REST API built using FastAPI
• Real-time prediction endpoint for transaction data
• Clean project structure following ML engineering practices

---

## Machine Learning Pipeline

The project follows a simple but realistic machine learning workflow:

1. Load the credit card transaction dataset
2. Separate features and target labels
3. Split the dataset into training and testing sets
4. Train a Random Forest classifier
5. Evaluate performance using standard classification metrics
6. Save the trained model to disk
7. Serve the model through an API for real-time predictions

Because fraud cases are rare in real datasets, the model uses class weighting to handle the imbalance between normal and fraudulent transactions.

---

## Model Evaluation

The model is evaluated using common classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score

For fraud detection systems, **recall is particularly important** because missing a fraudulent transaction can be costly.

---

## API Endpoint

The trained model is exposed through a FastAPI endpoint.

### Endpoint

POST `/predict`

### Example Request

```
{
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
```

### Example Response

```
{

"fraud_prediction": 0,
"fraud_probability": 0.29

}
```

The API returns both the predicted class and the probability score so that downstream systems can decide how to handle the transaction.

---

## How to Run the Project

Clone the repository:

```
git clone https://github.com/shaistabibi/fraud-detection-ml.git
```

Navigate into the project folder:

```
cd fraud-detection-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

## Train the model:

```
python src/train.py
```
## Testing Model

You can test predictions using:
```
python src/predict.py
```
Start the API server:

```
uvicorn app:app --reload
```

Once the server starts, the API will be available at:

```
http://127.0.0.1:8000
```

FastAPI will also automatically generate interactive API documentation at:

```
http://127.0.0.1:8000/docs
```

---

## Future Improvements

Possible improvements for this project include:

Adding feature engineering

Using advanced models (XGBoost, LightGBM)

Handling extreme class imbalance using SMOTE

Building a real-time streaming pipeline

Deploying the API to cloud platforms (AWS, GCP, Render)

Creating a dashboard for monitoring fraud predictions

---

## Dataset

This project uses the publicly available **Credit Card Fraud Detection dataset**.

The dataset contains anonymized transaction features and a binary label indicating whether the transaction was fraudulent.

---

## Author

Shaista Raziq
AI Engineer | Machine Learning | Generative AI

LinkedIn: https://www.linkedin.com/in/shaista-raziq-767064241/?originalSubdomain=pk
GitHub: https://github.com/shaistabibi
