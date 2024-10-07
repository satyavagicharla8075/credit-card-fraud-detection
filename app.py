import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template
# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features (X) and target variable (y)
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)


@app.route('/')
def hom():
    return render_template('home.html')

@app.route('/predict')
def predict():
    # Add code to render the detection page
    return render_template('predict.html')

@app.route('/about')
def about():
    # Add code to render the about page
    return render_template('about.html')

@app.route('/aboutus')
def about_us():
    # Add code to render the about us page
    return render_template('aboutus.html')

@app.route('/login')
def login():
    # Add code to render the login page
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def rj():
    # Get feature values from form
    features = [request.form[f'feature{i}'] for i in range(30)]
    
    # Convert non-empty feature values to float
    features = [float(x) if x.strip() else 0.0 for x in features]

    # Make prediction
    prediction = model.predict([features])[0]

    # Display result
    if prediction == 0:
        result = 'It is a Legitimate transaction'
    else:
        result = 'It is a Fraudulent transaction'

    return render_template('result.html', prediction_text=result )


if __name__ == '__main__':
    app.run(debug=True)