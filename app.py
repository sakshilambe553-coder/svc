import numpy as np
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load your SVM Model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Embedded HTML and CSS code
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Ownership Predictor</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #cbd5e1;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            width: 100%;
            max-width: 500px;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        .header {
            text-align: center;
            margin-bottom: 24px;
        }

        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        .header p {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .form-group {
            margin-bottom: 16px;
        }

        .form-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            background-color: #fff;
            transition: border-color 0.2s;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .btn-submit {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 12px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            transition: background-color 0.2s;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid;
        }

        .result-box.yes {
            background-color: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }

        .result-box.no {
            background-color: #fef2f2;
            border-color: #fecaca;
            color: #991b1b;
        }

        .result-box h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .result-box p {
            font-size: 1.4rem;
            font-weight: 700;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Laptop Ownership Predictor</h1>
        <p>Enter user details to predict laptop ownership</p>
    </div>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" min="1" max="100" placeholder="e.g. 25" required>
        </div>

        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="0">Female</option>
                <option value="1">Male</option>
            </select>
        </div>

        <div class="form-group">
            <label for="region">Region</label>
            <select id="region" name="region" required>
                <option value="0">City</option>
                <option value="1">Countryside</option>
            </select>
        </div>

        <div class="form-group">
            <label for="occupation">Occupation</label>
            <select id="occupation" name="occupation" required>
                <option value="0">Banker</option>
                <option value="1">Other</option>
                <option value="2">Student</option>
                <option value="3">Teacher</option>
            </select>
        </div>

        <div class="form-group">
            <label for="income">Income ($)</label>
            <input type="number" id="income" name="income" min="0" step="100" placeholder="e.g. 25000" required>
        </div>

        <button type="submit" class="btn-submit">Predict</button>
    </form>

    {% if prediction %}
    <div class="result-box {{ 'yes' if prediction == 'yes' else 'no' }}">
        <h3>Has Laptop?</h3>
        <p>{{ prediction.upper() }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form inputs in the model's required order: Age, Gender, Region, Occupation, Income
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    region = int(request.form['region'])
    occupation = int(request.form['occupation'])
    income = float(request.form['income'])
    
    # Format inputs into numpy array for the SVM model
    features = np.array([[age, gender, region, occupation, income]])
    
    # Generate prediction
    prediction = model.predict(features)[0]
    
    return render_template_string(HTML_TEMPLATE, prediction=str(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
