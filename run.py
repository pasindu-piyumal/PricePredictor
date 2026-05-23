from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import pickle
import keras

app = Flask(__name__, template_folder='app/templates')

# The fix is applied here:
model = keras.models.load_model('ml_models/model.h5', compile=False)
scaler = pickle.load(open('ml_models/min_max_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        longitude = request.form['longitude']
        latitude = request.form['latitude']
        age = request.form['age']
        rooms = request.form['rooms']
        bedrooms = request.form['bedrooms']
        population = request.form['population']
        households = request.form['households']
        income = request.form['income']
        proximity = request.form['proximity']

        proximity_mapping = {
            '<1H OCEAN': 0,
            'INLAND': 1,
            'ISLAND': 2,
            'NEAR BAY': 3,
            'NEAR OCEAN': 4
        }

        ocean_num = proximity_mapping[proximity]

        features = np.array([[float(longitude), float(latitude), float(age), float(rooms), float(bedrooms), float(population), float(households), float(income), float(ocean_num)]])

        features_scaled = scaler.transform(features)
        result = model.predict(features_scaled)
        final_price = round(float(result[0][0]), 2)

        user_inputs = {
            "Longitude": longitude,
            "Latitude": latitude,
            "Housing Age": age,
            "Total Rooms": rooms,
            "Total Bedrooms": bedrooms,
            "Population": population,
            "Households": households,
            "Median Income": income,
            "Ocean Proximity": proximity
        }
        return render_template('report.html', result=final_price, details=user_inputs)

    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)