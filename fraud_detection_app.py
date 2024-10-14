# fraud_detection_app.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from Google Drive
url = 'https://drive.google.com/uc?id=1wrXr_3skydYti-u7Hesxg3OhLuJ1fBPq'
credit_card_data = pd.read_csv(url)

# Preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
legit_sample = legit.sample(n=492, random_state=42)  # Set random_state for reproducibility
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Define features and target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning occurs
model.fit(X_train, Y_train)

# Streamlit user interface
st.title("Credit Card Fraud Detection")

# User input for prediction
st.sidebar.header("User Input Features")
def user_input_features():
    # Create a dictionary for the input features
    feature_values = {f'V{i}': st.sidebar.number_input(f"V{i}", value=0.0) for i in range(1, 29)}
    feature_values['Amount'] = st.sidebar.number_input("Amount", value=0.0)
    
    return pd.DataFrame(feature_values, index=[0])

input_df = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction and probabilities
    st.write("Prediction: ", "Fraud" if prediction[0] == 1 else "Legitimate")
    st.write("Prediction Probability: ", prediction_proba[0])

    # Optional: Display detailed probability breakdown
    st.write("Probability Breakdown:")
    for i, class_name in enumerate(["Legitimate", "Fraud"]):
        st.write(f"{class_name}: {prediction_proba[0][i]:.4f}")
