import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from WeightedBinaryCrossentropy import WeightedBinaryCrossentropy

data = pd.read_csv(r'..\Data\power_outage_data.csv')
data.drop(columns=['location'], inplace=True)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp')
X = data.loc[:,'temperature':'precipitation']
y = data.loc[:,'power_outage']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,shuffle=False,random_state=0)
X_train.reset_index(inplace=True, drop = True)
X_test.reset_index(inplace= True, drop = True)

def prepro(X):    
    scaler = MinMaxScaler()
    scaler.fit(X_train.loc[:,'temperature':])
    X_scaled = scaler.transform(X.loc[:,'temperature':])
    X_t = pd.DataFrame(X_scaled)
    return X_t.to_numpy()