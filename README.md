<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>Credit Card Fraud Detection</h1>
<p>This project involves building a machine learning model to detect fraudulent credit card transactions. The classifier is designed to predict whether a transaction is fraudulent or legitimate based on historical transaction data. The project utilizes data preprocessing techniques, a Logistic Regression model, and a label encoder for this task.</p>

<h2>Features</h2>
<ul>
    <li><strong>Fraud Detection:</strong> Predicts whether a given credit card transaction is fraudulent or legitimate.</li>
    <li><strong>Machine Learning Model:</strong> A Logistic Regression model trained on a highly imbalanced dataset to achieve high accuracy in detecting frauds.</li>
    <li><strong>User-Friendly Interface:</strong> A web application built with Streamlit that allows users to input transaction data and receive fraud detection results in real-time.</li>
    <li><strong>Data Handling:</strong> Efficient data preprocessing and balancing techniques applied to the imbalanced dataset for better model performance.</li>
</ul>

<h2>Project Components</h2>

<h3>Data Preprocessing</h3>
<p>The dataset used is highly imbalanced, with only 0.172% of the transactions classified as fraud. To address this, the project implements data sampling techniques such as undersampling to balance the dataset and improve the model's performance.</p>

<h3>Feature Extraction</h3>
<p>Extracted key features from the transaction data such as transaction amount, time, and other behavioral characteristics. These features are used as inputs to the machine learning model to detect fraudulent patterns in the transactions.</p>

<h3>Model Training and Comparison</h3>
<ul>
    <li>Tested different machine learning models (Logistic Regression, Decision Trees, etc.) on the preprocessed dataset.</li>
    <li>Compared performance metrics such as accuracy, precision, recall, and F1-score to select the best-performing model for fraud detection.</li>
    <li>Selected Logistic Regression as the final model for further refinement and deployment.</li>
</ul>

<h3>Logistic Regression Model Training</h3>
<p>The Logistic Regression model was trained on the balanced dataset and evaluated using various metrics like accuracy and F1-score. This model was saved as <code>final_model.pkl</code> for deployment purposes.</p>

<h3>Streamlit Application</h3>
<ul>
    <li>Developed a web application using Streamlit to provide an interactive interface for fraud detection.</li>
    <li>Allows users to input transaction details and view real-time predictions of whether the transaction is fraudulent or legitimate.</li>
    <li>Displays fraud detection probabilities for better understanding of model confidence.</li>
</ul>

<h3>Error Handling</h3>
<ul>
    <li>Handled cases where the model or the preprocessed dataset fails to load with appropriate error messages.</li>
    <li>Validated transaction inputs to ensure they are properly formatted.</li>
    <li>Provided user-friendly error messages if inputs are incorrect or other issues arise during prediction.</li>
</ul>

<h2>Error Handling and Messages</h2>
<ul>
    <li>If the model or dataset fails to load, the app displays an appropriate error message and stops further execution.</li>
    <li>If the input transaction data is incomplete or incorrect, the user is prompted to correct the data.</li>
    <li>Any other errors during prediction are caught and presented to the user, ensuring a seamless experience.</li>
</ul>

<h2>Conclusion</h2>
<p>This project demonstrates the application of machine learning for credit card fraud detection. By leveraging a Logistic Regression model and an easy-to-use Streamlit interface, users can quickly predict whether a transaction is fraudulent. The use of effective data preprocessing and robust error handling ensures the reliability and accuracy of the system.</p>

<h2>Live Demo</h2>
<p>You can check the deployed version of the project on Streamlit using the following link:</p>
<p><a href="https://codtech-task-1-nivfxnyn4vpcmsssp8jylq.streamlit.app/" target="_blank">Live Demo</a></p>

</body>
</html>
