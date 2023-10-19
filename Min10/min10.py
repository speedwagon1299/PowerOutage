import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Generate data with more realistic weather patterns for Mangalore
data = []

# Generate data for a single day at 10-minute intervals
start_time = datetime(2023, 10, 1, 0, 0)
num_intervals = 14400

for _ in range(num_intervals):
    city = "Mangalore"  # Set the city to Mangalore
    timestamp = start_time
    hour_progress = start_time.hour + start_time.minute / 60.0

    # Create more realistic temperature pattern for Mangalore
    temperature = round(30 + 4 * np.sin(2 * np.pi * 0.02 * hour_progress), 1)

    # Create more realistic humidity pattern for Mangalore
    humidity = round(75 + 15 * np.sin(2 * np.pi * 0.02 * hour_progress), 1)

    # Create more realistic wind speed pattern for Mangalore
    wind_speed = round(10 + 4 * np.sin(2 * np.pi * 0.04 * hour_progress), 1)

    # Create more realistic precipitation pattern for Mangalore
    precipitation = abs(round(10 * np.sin(2 * np.pi * 0.04 * hour_progress), 1))

    # Determine if a power outage occurs based on weather conditions
    if (
        (hour_progress >= 6 and hour_progress < 8) or
        (hour_progress >= 12 and hour_progress < 13) or
        (hour_progress >= 18 and hour_progress < 20)
    ):
        temperature = round(36.0 + 3*np.random.random())
        humidity = round(55.0 + 3*np.random.random())
        wind_speed = round(5.0 + 2*np.random.random())
        precipitation = round(2*np.random.random())
        outage_occurred = 1
    else:
        outage_occurred = 0

    data.append([timestamp, city, temperature, humidity, wind_speed, precipitation, outage_occurred])

    start_time += timedelta(minutes=10)

# Create a DataFrame
df = pd.DataFrame(data, columns=['timestamp', 'location', 'temperature', 'humidity', 'wind_speed', 'precipitation', 'power_outage'])

# Remove duplicates
df = df.drop_duplicates()

# Save the DataFrame as a CSV file
df.to_csv('multivariate_timeseries_data_mangalore_extreme.csv', index=False)
