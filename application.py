from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__)

# Load the pickled scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ridge.pkl', 'rb') as f:
    model = pickle.load(f)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST','GET'])
def predict():
    # Get input values from the 
    
    Temperature = float(request.form['Temperature'])
    RH = float(request.form['RH'])
    Ws = float(request.form['Ws'])
    Rain = float(request.form['Rain'])
    FFMC = float(request.form['FFMC'])
    ISI = float(request.form['ISI'])
    BUI = float(request.form['BUI'])
    Classes = float(request.form['Classes'])
    Region = float(request.form['Region'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, ISI, BUI, Classes, Region]],
                               columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'ISI', 'BUI', 'Classes', 'Region'])
    

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)[0]
    

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    application.run(debug=True)