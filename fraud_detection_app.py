import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Streamlit user interface
st.title("Credit Card Fraud Detection")

# Allow the user to upload their own dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    credit_card_data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

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
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        # Save the model to a file
        dump(model, 'logistic_regression_model.joblib')  # Save your trained model
        st.success("Model trained and saved successfully!")

        # Now allow the user to upload a dataset for predictions
        st.sidebar.header("Upload Your Dataset for Prediction")
        user_uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], key="user_upload")

        if user_uploaded_file is not None:
            user_data = pd.read_csv(user_uploaded_file)
            
            # Check if the uploaded dataset has the same columns as the model's training dataset
            if 'Class' in user_data.columns:
                user_data.drop(columns='Class', axis=1, inplace=True)  # Remove Class column if exists

            if user_data.shape[1] == X.shape[1]:  # Check if the feature columns match
                # Load the pre-trained model
                model = load('logistic_regression_model.joblib')

                # Make predictions
                predictions = model.predict(user_data)
                prediction_proba = model.predict_proba(user_data)

                # Display results
                result_df = user_data.copy()
                result_df['Prediction'] = ["Fraud" if pred == 1 else "Legitimate" for pred in predictions]
                result_df['Prediction Probability'] = [f"{prob[1]:.2f}" for prob in prediction_proba]

                st.write("Prediction Results:")
                st.dataframe(result_df)
            else:
                st.error("The uploaded dataset does not have the required number of features.")
else:
    st.error("Please upload a dataset to proceed.")
