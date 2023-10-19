import csv
import pandas as pd
from datetime import datetime, timedelta
import math
import random

# Function to determine power outage based on past features
def determine_power_outage(temperature, humidity, wind_speed, precipitation):
    if (temperature >= 28 and humidity >= 92) or (wind_speed >= 2 and precipitation >= 2):
        return 1
    else:
        return 0

# Generate data with cyclic features with added randomness
start_date = datetime(2022, 5, 20, 0, 0)
end_date = datetime(2023, 5, 24, 0, 0)  # You can change the end date as needed
current_date = start_date
data = []
past_features = []  # List to store past feature sets

while current_date <= end_date:
    # Define cyclic formulas for features with added randomness
    time_step = (current_date - start_date).total_seconds() / 3600  # Time in hours

    # Introduce randomness by adding a small value sampled from a normal distribution

    temperature = round(25 + 5 * math.sin(2 * math.pi * (time_step % 24) / 24) + random.normalvariate(0, 0.5),2)
    humidity = round(80 + 17 * math.sin(2 * math.pi * (time_step % 24) / 24) + random.normalvariate(0, 0.5),2)
    wind_speed = abs(round(3.3* math.sin(2 * math.pi * (time_step % 24) / 24) + random.normalvariate(0, 0.2),3))
    precipitation = round(abs(4 * math.sin(2 * math.pi * (time_step % 24) / 24) + random.normalvariate(0, 0.2))/10,2)

    # Append the current feature set to the list of past features
    past_features.append([temperature, humidity, wind_speed, precipitation])

    # Ensure that past_features contains at most the last 4 sets
    if len(past_features) > 4:
        past_features.pop(0)

    # Determine power outage based on the past 4 sets of features
    power_outage = determine_power_outage(*past_features[-1])

    data.append([
        current_date.strftime('%Y-%m-%d %H:%M:%S'),
        'Mangalore',
        round(temperature, 2),
        humidity,
        round(wind_speed, 2),
        round(precipitation, 2),
        power_outage
    ])

    current_date += timedelta(hours=1)

# Create a Pandas DataFrame from the generated data
df = pd.DataFrame(data, columns=['timestamp', 'location', 'temperature', 'humidity', 'wind_speed', 'precipitation', 'power_outage'])

# Save the data to a CSV file
df.to_csv('power_outage_data.csv', index=False)
