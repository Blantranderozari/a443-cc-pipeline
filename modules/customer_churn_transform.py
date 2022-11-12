import tensorflow as tf 
import tensorflow_transform as tft 

CATEGORICAL_FEATURES = {
    "InternetService": 3,
    "SeniorCitizen": 2,
    "PaperlessBilling": 2,
    "Partner": 2,
    "PhoneService": 2,
    "StreamingTV": 3,
    "gender":2
}

NUMERICAL_FEATURES = [
    "MonthlyCharges",
    "TotalCharges",
    "tenure"
]

LABEL_KEY = "Churn"