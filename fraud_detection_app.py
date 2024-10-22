import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from joblib import load

# Streamlit user interface
st.title("Credit Card Fraud Detection")

# Explanation of the app
st.sidebar.header("How It Works")
st.sidebar.write("""
This app allows you to detect fraudulent transactions in a credit card dataset.
1. **Upload your dataset**: Make sure it contains relevant features for transaction analysis.
2. The app will determine how many transactions are fraudulent and how many are legitimate.
""")

# Allow the user to upload their own dataset for predictions
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    credit_card_data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Preprocessing
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]

    # Count legitimate and fraudulent transactions
    count_legit = legit.shape[0]
    count_fraud = fraud.shape[0]

    # Display results
    st.write(f"**Total Legitimate Transactions:** {count_legit}")
    st.write(f"**Total Fraudulent Transactions:** {count_fraud}")

    # Load the pre-trained model
    model = load('logistic_regression_model.joblib')

    # Make predictions on the uploaded dataset
    predictions = model.predict(credit_card_data)

    # Add predictions to the dataframe
    credit_card_data['Prediction'] = ["Fraud" if pred == 1 else "Legitimate" for pred in predictions]

    # Optionally show the updated dataframe with predictions
    st.write("Prediction Results:")
    st.dataframe(credit_card_data)

else:
    st.error("Please upload a dataset to proceed.")
