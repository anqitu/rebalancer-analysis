import urllib.request
import json
from datetime import datetime
import pandas as pd
pd.options.display.max_columns = 999

# api_call = 'https://api.darksky.net/forecast/3797bfa568cb62217f3a3dcf22516d4e/51.5074,0.1278,'
#
# for unix_time in range(1514764800, 1514764800 + 86400*365, 86400):
#     with urllib.request.urlopen("https://api.darksky.net/forecast/3797bfa568cb62217f3a3dcf22516d4e/51.5074,0.1278," + str(unix_time)) as url:
#         data = json.loads(url.read().decode())
#
#     with open('data/raw/london-weather/{}.json'.format(unix_time), 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

daily_data = []
hourly_data = []

for unix_time in range(1514764800, 1514764800 + 86400*365, 86400):
    with open('data/raw/london-weather/{}.json'.format(unix_time)) as f:
        data = json.load(f)

    daily_data += data['daily']['data']
    hourly_data += data['hourly']['data']


daily_data_df = pd.DataFrame(data=daily_data)
daily_data_df.describe(include='all')
daily_data_df['time'].max()
daily_data_df.info()
daily_data_df.isnull().sum()
daily_data_df[daily_data_df.isnull().any(axis=1)]
daily_data_df['icon'].value_counts()


daily_data_df['time'] = pd.to_datetime(daily_data_df['time'],unit='s')
daily_data_df = daily_data_df.fillna(method = 'ffill').fillna(method = 'bfill')
daily_data_df['icon'] = daily_data_df['icon'].replace('partly-cloudy-day', 'cloudy').replace('clear-day', 'clear')
daily_data_df['rain'] = (daily_data_df['icon'] == 'rain').astype(int)
daily_data_df['clear'] = (daily_data_df['icon'] == 'clear').astype(int)
daily_data_df['cloudy'] = (daily_data_df['icon'] == 'cloudy').astype(int)
daily_data_df = daily_data_df[['time', 'rain', 'clear', 'cloudy', 'apparentTemperatureHigh', 'apparentTemperatureLow', 'precipIntensity', 'dewPoint', 'humidity', 'windSpeed', 'uvIndex', 'visibility']]
daily_data_df = daily_data_df.rename(columns = {'time': 'Time'})

hourly_data_df = pd.DataFrame(data=hourly_data)
hourly_data_df.describe(include='all')
hourly_data_df['time'].max()
hourly_data_df.info()
hourly_data_df.isnull().sum()
hourly_data_df['icon'].value_counts()
hourly_data_df[hourly_data_df.isnull().any(axis=1)]


hourly_data_df['time'] = pd.to_datetime(hourly_data_df['time'],unit='s')
hourly_data_df = hourly_data_df.fillna(method = 'ffill')
hourly_data_df['icon'] = hourly_data_df['icon'].replace('partly-cloudy-day', 'cloudy').replace('partly-cloudy-night', 'cloudy').replace('clear-day', 'clear').replace('clear-night', 'clear')
hourly_data_df['rain'] = (hourly_data_df['icon'] == 'rain').astype(int)
hourly_data_df['clear'] = (hourly_data_df['icon'] == 'clear').astype(int)
hourly_data_df['cloudy'] = (hourly_data_df['icon'] == 'cloudy').astype(int)
hourly_data_df = hourly_data_df[['time', 'rain', 'clear', 'cloudy', 'apparentTemperature', 'precipIntensity', 'dewPoint', 'humidity', 'windSpeed', 'uvIndex', 'visibility']]
hourly_data_df = hourly_data_df.rename(columns = {'time': 'Time'})

daily_data_df.to_csv('data/processed/london_daily_weather.csv', index = False)
hourly_data_df.to_csv('data/processed/london_hourly_weather.csv', index = False)
