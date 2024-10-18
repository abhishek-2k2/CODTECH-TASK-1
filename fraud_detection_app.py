import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from Google Drive
url = 'https://drive.google.com/uc?id=1wrXr_3skydYti-u7Hesxg3OhLuJ1fBPq'
credit_card_data = pd.read_csv(url)

# Check and display the columns in the dataset
st.write("Dataset Columns:", credit_card_data.columns.tolist())

# Check if the 'Class' column exists
if 'Class' not in credit_card_data.columns:
    st.error("The dataset does not contain a 'Class' column.")
else:
    # Preprocessing
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    # Sample the legitimate transactions for balance
    legit_sample = legit.sample(n=492, random_state=42)  # Set random_state for reproducibility
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    # Provide the option to download the preprocessed dataset
    csv_data = new_dataset.to_csv(index=False)
    st.download_button(
        label="Download Preprocessed Dataset as CSV",
        data=csv_data,
        file_name='preprocessed_creditcard_fraud.csv',
        mime='text/csv'
    )

    # Define features and target
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Streamlit user interface
    st.title("Credit Card Fraud Detection")

    # User input for prediction
    st.sidebar.header("User Input Features")
    
    def user_input_features():
        # Adjust the following according to your dataset's feature columns
        V1 = st.sidebar.number_input("V1", value=0.0)
        V2 = st.sidebar.number_input("V2", value=0.0)
        V3 = st.sidebar.number_input("V3", value=0.0)
        V4 = st.sidebar.number_input("V4", value=0.0)
        V5 = st.sidebar.number_input("V5", value=0.0)
        V6 = st.sidebar.number_input("V6", value=0.0)
        V7 = st.sidebar.number_input("V7", value=0.0)
        V8 = st.sidebar.number_input("V8", value=0.0)
        V9 = st.sidebar.number_input("V9", value=0.0)
        V10 = st.sidebar.number_input("V10", value=0.0)
        V11 = st.sidebar.number_input("V11", value=0.0)
        V12 = st.sidebar.number_input("V12", value=0.0)
        V13 = st.sidebar.number_input("V13", value=0.0)
        V14 = st.sidebar.number_input("V14", value=0.0)
        V15 = st.sidebar.number_input("V15", value=0.0)
        V16 = st.sidebar.number_input("V16", value=0.0)
        V17 = st.sidebar.number_input("V17", value=0.0)
        V18 = st.sidebar.number_input("V18", value=0.0)
        V19 = st.sidebar.number_input("V19", value=0.0)
        V20 = st.sidebar.number_input("V20", value=0.0)
        V21 = st.sidebar.number_input("V21", value=0.0)
        V22 = st.sidebar.number_input("V22", value=0.0)
        V23 = st.sidebar.number_input("V23", value=0.0)
        V24 = st.sidebar.number_input("V24", value=0.0)
        V25 = st.sidebar.number_input("V25", value=0.0)
        V26 = st.sidebar.number_input("V26", value=0.0)
        V27 = st.sidebar.number_input("V27", value=0.0)
        V28 = st.sidebar.number_input("V28", value=0.0)
        Amount = st.sidebar.number_input("Amount", value=0.0)

        data = {
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'V7': V7,
            'V8': V8,
            'V9': V9,
            'V10': V10,
            'V11': V11,
            'V12': V12,
            'V13': V13,
            'V14': V14,
            'V15': V15,
            'V16': V16,
            'V17': V17,
            'V18': V18,
            'V19': V19,
            'V20': V20,
            'V21': V21,
            'V22': V22,
            'V23': V23,
            'V24': V24,
            'V25': V25,
            'V26': V26,
            'V27': V27,
            'V28': V28,
            'Amount': Amount
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.write("Prediction: ", "Fraud" if prediction[0] == 1 else "Legitimate")
        st.write("Prediction Probability: ", prediction_proba[0])
