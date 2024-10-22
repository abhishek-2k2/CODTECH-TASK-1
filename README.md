<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>

<!-- Project Information -->
<section>
    <h2>Project Information</h2>
    <p><strong>Name:</strong> Abhishek Chauhan</p>
    <p><strong>Company:</strong> Codtech IT Solutions</p>
    <p><strong>ID:</strong> CT6WDS1850</p>
    <p><strong>Domain:</strong> Machine Learning</p>
    <p><strong>Duration:</strong> September 5, 2024 to October 20, 2024</p>
</section>

<h1>Credit Card Fraud Detection</h1>
<p>This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project can be found <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" download>here</a>.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>Credit card fraud is a significant problem in the financial sector. This project utilizes a machine learning approach to detect fraudulent transactions from a dataset of credit card transactions. The model employed is a Logistic Regression classifier, and the project demonstrates data preprocessing, model training, and evaluation.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.</p>
<p>Dataset link: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" download>Credit Card Fraud Detection</a></p>

<h2 id="installation">Installation</h2>
<p>Clone the repository:</p>
<pre><code>
git clone https://github.com/abhishek-2k2/credit-card-fraud-detection.git
cd credit-card-fraud-detection
</code></pre>
<p>Install the required packages:</p>
<pre><code>
pip install -r requirements.txt
</code></pre>

<h2 id="usage">Usage</h2>
<p>Load the dataset:</p>
<pre><code>
import pandas as pd
credit_card_data = pd.read_csv('path_to_your_dataset/credit_data.csv')
</code></pre>
<p>Preprocess the data and train the model:</p>
<pre><code>
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data:', test_data_accuracy)
</code></pre>

<h2 id="results">Results</h2>
<p>The Logistic Regression model achieved an accuracy score of <code>test_data_accuracy</code> on the test dataset. This demonstrates the model's ability to distinguish between legitimate and fraudulent transactions with a high degree of accuracy.</p>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please feel free to submit a Pull Request.</p>

</body>
</html>
