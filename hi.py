'''
api_key = '2a8be53043c3523cb8f9e00f4843aa8c'
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/location', methods=['POST'])
def receive_location():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']

    # Print the received latitude and longitude
    print(f'Latitude: {latitude}, Longitude: {longitude}')

    return 'Location received successfully!', 200

import requests
import json
import csv

@app.route('/')
def index():
    city_name = "mangalore"

    # Get the city's coordinates (lat and lon)
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    req = requests.get(url)
    data = req.json()

    name = data['name']

    lon = data['coord']['lon']
    lat = data['coord']['lat']

    url2 = f'https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&appid={api_key}'
    req2 = requests.get(url2)
    data2 = req2.json()

    temp = round(((data2['list'][0]['main']['temp_min'] + data2['list'][0]['main']['temp_max'])/2) - 273,2)
    humid = data2['list'][0]['main']['humidity']
    wind = round(data2['list'][0]['wind']['speed']*10,2)

    extracted_data = []
    count = 0
    for item in data2['list']:
        count = count + 1
        dt = item['dt_txt']
        temp_min = item['main']['temp_min']
        temp_max = item['main']['temp_max']
        humidity = item['main']['humidity']
        precipitation = item.get('rain', {}).get('1h', 0)
        wind_speed = item['wind']['speed']

        avg_temp = (temp_min + temp_max)/2
        extracted_data.append({
            'timestamp': dt,
            'temperature': round(avg_temp - 273, 2),
            'humidity': round(humidity, 2),
            'precipitation': round(precipitation, 2),
            'wind_speed': round(wind_speed, 2),
            'location': 'Mangalore'
        })
        if count == 5:
            break

    csv_filename = 'weather_data_new.csv'
    csv_headers = ['timestamp', 'location', 'temperature', 'humidity', 'wind_speed', 'precipitation']

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(extracted_data)

    return render_template('index.html', name=name, lon=lon, lat=lat, temp=temp, humid=humid, wind=wind)


if __name__ == '__main__':
    app.run(debug=True)
'''

from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def prepro(X):    
    scaler = MinMaxScaler()
    scaler.fit(X_train.loc[:,'temperature':])
    X_scaled = scaler.transform(X.loc[:,'temperature':])
    X_t = pd.DataFrame(X_scaled)
    return X_t.to_numpy()

def load_model_and_predict():
    model = tfk.Sequential()
    model.add(tfk.Input(shape = (4,4)))
    model.add(tfk.layers.Bidirectional(tfk.layers.LSTM(128, return_sequences=True)))
    model.add(tfk.layers.ReLU())
    model.add(tfk.layers.Dropout(0.3))
    model.add(tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True)))
    model.add(tfk.layers.ReLU())
    model.add(tfk.layers.Dropout(0.3))
    model.add(tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True)))
    model.add(tfk.layers.ReLU())
    model.add(tfk.layers.Dropout(0.3))
    model.add(tfk.layers.LSTM(128, return_sequences=False))
    model.add(tfk.layers.ReLU())
    model.add(tfk.layers.Dense(1))
    
    model.load_weights(r'Pretrained Weights\po_hour')

    data = pd.read_csv(r'weather_data_new.csv')
    data.drop(columns=['location'], inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp')

    X = prepro(data)
    gen = tfk.preprocessing.sequence.TimeseriesGenerator(np.asarray(X), np.asarray(X), length=4, sampling_rate=1, stride=1, batch_size=32)
    yhat = model.predict(gen)
    yhat = [1 if x > 0.5 else 0 for x in yhat]

    if yhat[0] == 0:
        print('NO DETECTABLE POWER OUTAGE IN THE COMING HOUR')
    else:
        print('POSSIBILITY OF POWER OUTAGE IN THE COMING HOUR')

@app.route('/location', methods=['POST'])
def receive_location():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']

    # Print the received latitude and longitude
    print(f'Latitude: {latitude}, Longitude: {longitude}')

    return 'Location received successfully!', 200

# Add your remaining routes and code here...

if __name__ == '__main__':
    app.run(debug=True)
