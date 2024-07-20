**Detects power outage utilising (user generated) features of timestamp (10 minute intervals) , location(mangalore), temperature, humidity, wind_speed, precipitation, power_outage using LSTM
(All features based on weather provided by embedded weather app)**


----------------------------------------------------------------------------------------------------------------------------------------------------


**Model/LSTM_hour.ipynb**

Main Model to train data and evaluate test data

Loss vs Epoch

ROC AUC

Confusion Matrix + F1 Measure

Precision-Recall


----------------------------------------------------------------------------------------------------------------------------------------------------


**Data/hourly.py**

Dataset geenrated by python code

All features (except power_outage and timestamp) trignometric function of timestamp and other feature

Power outage decided by extreme conditions occuring at certain timings

CSV saved as **power_outage_data.csv**


----------------------------------------------------------------------------------------------------------------------------------------------------


**Model/WeightedBinaryCrossentropy.py**

In the model, due to increased possibility of non power outage (0), there is a heavy class imbalance of 0s to 1s as present in real world data.

Tensorflow code to modify weights in binary cross entropy, function used during model compilation

Array[0] - Increase => Prevent 1s appearing instead of 0s => Reduces chances of False Positives

Array[1] - Increase => Prevent 0s appearing instead of 1s => Reduces chances of False Negatives


----------------------------------------------------------------------------------------------------------------------------------------------------


**Pretrained Weights**

Current trained weights saved

Use model.load_weights(r'../Pretrained Weights/po_hour') to load the weights (from actual.ipynb)


----------------------------------------------------------------------------------------------------------------------------------------------------

```
Sequential Model:

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bidirectional_3 (Bidirectio  (None, 4, 256)           136192    
 nal)                                                            
                                                                 
 re_lu_4 (ReLU)              (None, 4, 256)            0         
                                                                 
 dropout_3 (Dropout)         (None, 4, 256)            0         
                                                                 
 bidirectional_4 (Bidirectio  (None, 4, 512)           1050624   
 nal)                                                            
                                                                 
 re_lu_5 (ReLU)              (None, 4, 512)            0         
                                                                 
 dropout_4 (Dropout)         (None, 4, 512)            0         
                                                                 
 bidirectional_5 (Bidirectio  (None, 4, 512)           1574912   
 nal)                                                            
                                                                 
 re_lu_6 (ReLU)              (None, 4, 512)            0         
                                                                 
 dropout_5 (Dropout)         (None, 4, 512)            0         
                                                                 
 lstm_7 (LSTM)               (None, 128)               328192    
                                                                 
 re_lu_7 (ReLU)              (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 3,090,049
Trainable params: 3,090,049
Non-trainable params: 0
_________________________________________________________________
```


F1 measure: 0.8536585365853657

ROC AUC: 0.9461077844311377


![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/58c86a3f-2ffd-4216-961b-d62a4a5fa106)

![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/222b8c44-a6f6-40dd-845e-b208df7b2571)

![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/a15508ab-ea16-468e-83c1-d2c07f2b0561)


----------------------------------------------------------------------------------------------------------------------------------------------------


**Model/fun.py**

To preprocess input in **actual.ipynb**


----------------------------------------------------------------------------------------------------------------------------------------------------


**DataRetrieval/WeatherAPI.ipynb**

To retrieve past 5 hours data on aforementioned features from openweathermap.org api student key

**weather_data_new.csv** created


----------------------------------------------------------------------------------------------------------------------------------------------------


**Model/actual.ipynb**

To access data from **DataRetrieval/weather_data_new.csv** and predict possibility of power outage


----------------------------------------------------------------------------------------------------------------------------------------------------


(EXTRA: Min10 is for the 10 minute interval power outage prediction model, but due to lack of reliable data, hourly interval based lstm built)

