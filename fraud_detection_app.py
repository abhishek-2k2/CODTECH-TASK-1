import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from joblib import load
import matplotlib.pyplot as plt

# Streamlit user interface
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection App")

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

    # Display basic information about the dataset
    st.subheader("Dataset Overview")
    st.write(credit_card_data.head())
    st.write(f"Total Transactions: {len(credit_card_data)}")

    # Preprocessing to count legitimate and fraudulent transactions
    if 'Class' in credit_card_data.columns:
        legit = credit_card_data[credit_card_data.Class == 0]
        fraud = credit_card_data[credit_card_data.Class == 1]
    else:
        legit = credit_card_data
        fraud = pd.DataFrame()

    count_legit = legit.shape[0]
    count_fraud = fraud.shape[0]

    # Display the counts
    st.write(f"**Total Legitimate Transactions:** {count_legit}")
    st.write(f"**Total Fraudulent Transactions:** {count_fraud}")

    # Visualization
    st.subheader("Transaction Distribution")
    labels = ['Legitimate', 'Fraudulent']
    sizes = [count_legit, count_fraud]
    colors = ['#4CAF50', '#FF5733']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    st.pyplot(fig)

    # Load the pre-trained model
    model = load('logistic_regression_model.joblib')

    # Make predictions on the relevant features
    feature_columns = credit_card_data.columns.drop('Class', errors='ignore')  # Exclude 'Class' if it exists
    user_data = credit_card_data[feature_columns]
    predictions = model.predict(user_data)

    # Add predictions to the dataframe
    credit_card_data['Prediction'] = ["Fraud" if pred == 1 else "Legitimate" for pred in predictions]

    # Show prediction results in a table
    st.subheader("Prediction Results")
    result_df = credit_card_data[['Prediction']].copy()
    st.dataframe(result_df)

    # Count the predictions
    prediction_count = result_df['Prediction'].value_counts()
    st.write(f"**Predicted Legitimate Transactions:** {prediction_count.get('Legitimate', 0)}")
    st.write(f"**Predicted Fraudulent Transactions:** {prediction_count.get('Fraud', 0)}")

else:
    st.error("Please upload a dataset to proceed.")
