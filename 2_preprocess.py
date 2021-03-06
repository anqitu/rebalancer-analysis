import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import json

# START_DATE = datetime(year = 2018, month = 1, day = 1, hour = 0)
#
# stations = pd.read_csv('data/raw/stations.csv')
# stations.describe()
# len(stations['Station ID'].unique())
#
# journeys_df = pd.read_csv('data/processed/london_clean_journeys.csv',
#     parse_dates=['Start Date', 'End Date'], infer_datetime_format = True)
# journeys_df.shape[0]
# # print(journeys_df[(journeys_df['End Date'] - journeys_df['Start Date']).astype('timedelta64[m]').astype(int) * 60 != journeys_df['Duration']].shape[0])
# # journeys_df.head()
# # journeys_df.info()
# # journeys_df.describe()
#
# # group journeys by Station ID and by set window intervals
# def get_time_groups(grouper_key, granularity, groupby):
#     grouper = pd.Grouper(key=grouper_key, freq=str(granularity) + 'Min', label='right')
#     groups = journeys_df.groupby([groupby, grouper]).size()
#     groups = groups.unstack(fill_value=0).stack() # fill nonexistent counts as 0
#     return groups.reset_index()
#
# hours = 2
# granularity = 60 * hours # minutes
#
# journeys_count_df = get_time_groups('Start Date', granularity, 'StartStation Id')
# journeys_count_df = journeys_count_df.rename(columns={0: 'Out', 'StartStation Id': 'Station ID', 'Start Date': 'Time'})
# journeys_count_df['In'] = get_time_groups('End Date', granularity, 'EndStation Id')[0]
# journeys_count_df['Time'] = journeys_count_df['Time'] - pd.Timedelta(hours=hours)
# journeys_count_df['In'] = journeys_count_df['In'].fillna(0).astype(int)
# journeys_count_df['Out'] = journeys_count_df['Out'].fillna(0).astype(int)
# # journeys_count_df.head(10)
# # journeys_count_df.describe()
#
# journeys_count_df['Time'].min()
# journeys_count_df['Time'].max()
# existing_tiemsteps = pd.to_datetime(list(journeys_count_df['Time'].unique()))
# len(existing_tiemsteps)
# sorted(existing_tiemsteps)
# all_timesteps = list(pd.date_range(START_DATE, periods=12*365, freq=str(granularity) + 'Min'))
# len(all_timesteps)
#
# journeys_count_df.shape[0] # 3495240
#
# missing_timesteps = set(all_timesteps).difference(set(existing_tiemsteps))
# for timestep in missing_timesteps:
#     station_counts = len(journeys_count_df['Station ID'].unique())
#     missing_df = pd.DataFrame(data = {'Station ID': journeys_count_df['Station ID'].unique(),
#                                       'Time': [timestep] * station_counts,
#                                       'In': [0] * station_counts,
#                                       'Out': [0] * station_counts})
#     journeys_count_df = journeys_count_df.append(missing_df)
# journeys_count_df.shape[0] # 3495240
#
# # Remove stations with no journeys
# stations_no_journey = set(stations['Station ID'].unique()).difference(set(journeys_count_df['Station ID'].unique()))
# stations = stations[~stations['Station ID'].isin(stations_no_journey)]
# len(stations['Station ID'].unique())
#
# # Export to json
# stations_dict = []
# for index, row in stations.iterrows():
#     station = {}
#     station['name'] = row['Station Name']
#     station['id'] = row['Station ID']
#     station['capacity'] = row['Capacity']
#     station['coordinates'] = [row['Longitude'], row['Latitude']]
#     stations_dict.append(station)
# with open('data/processed/london_stations.json', 'w', encoding='utf-8') as f:
#     json.dump(stations_dict, f, ensure_ascii=False, indent=4)
#
# journeys_count_df = journeys_count_df[journeys_count_df['Station ID'].isin(stations['Station ID'])]
# journeys_count_df.shape[0] # 3385740
# journeys_count_df = journeys_count_df.sort_values(['Station ID', 'Time'])
# journeys_count_df.to_csv('data/processed/london_journeys_count_with_{}h_interval.csv'.format(hours), index=False)
# journeys_count_df.head(50)


# Average Station Hourly Demand
journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
dayofweek_mapper = dict(enumerate(list(calendar.day_name)))
journeys_count_df['DayOfWeek'] = journeys_count_df['Time'].dt.dayofweek.map(dayofweek_mapper)
journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
journeys_count_df['Week Hour'] = journeys_count_df['Time'].dt.dayofweek * 24 + journeys_count_df['Time'].dt.hour
# journeys_count_df.head(50)
weeks = [g for n, g in journeys_count_df.groupby(pd.Grouper(key='Time', freq='W'))]
len(weeks)

fig = plt.figure(figsize=(15, 8))
plt.title('Average Station 2-hour Interval Demand by Week', size=25, pad=20)
colors = sns.cubehelix_palette(8)
for i, week in enumerate(weeks[1:9]):
    sns.lineplot(week['Week Hour'], week['Out'], color=colors[i], ci=None, label="Week {}".format(i+1))
plt.xlabel('DayOfWeek', size=20)
plt.ylabel('Trips', size=20)
plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
plt.yticks(size=15)
plt.ylim(0)
plt.legend(fontsize=15)
# plt.legend(fontsize=15, bbox_to_anchor = (1.15, 1))
fig.savefig('images/avg_station_hourly_demand', dpi = 200, bbox_inches = 'tight')

fig = plt.figure(figsize=(15, 8))
plt.title('Average Station 2-hour Interval Supply by Week', size=25, pad=20)
colors = sns.cubehelix_palette(8)
for i, week in enumerate(weeks[30:38]):
    sns.lineplot(week['Week Hour'], week['In'], color=colors[i], ci=None, label="Week {}".format(i+1))
plt.xlabel('DayOfWeek', size=20)
plt.ylabel('Trips', size=20)
plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
plt.yticks(size=15)
plt.ylim(0)
plt.legend(fontsize=15)
# plt.legend(fontsize=15, bbox_to_anchor = (1.15, 1))
fig.savefig('images/avg_station_hourly_supply', dpi = 200, bbox_inches = 'tight')

# fig = plt.figure(figsize=(15, 8))
# sns.lineplot(journeys_count_df.iloc[:10000]['Week Hour'], journeys_count_df.iloc[:10000]['Out'])
# plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
#
# fig = plt.figure(figsize=(15, 8))
# sns.lineplot(journeys_count_df.iloc[:10000]['Week Hour'], journeys_count_df.iloc[:10000]['In'])
# plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))


daily_data_df = pd.read_csv('data/processed/london_daily_weather.csv', parse_dates=['Time'])
hourly_data_df = pd.read_csv('data/processed/london_hourly_weather.csv', parse_dates=['Time'])
daily_data_df.shape
hourly_data_df.shape


combined = journeys_count_df.merge(hourly_data_df)
combined.info()
combined.corr()['Out'].apply(abs).sort_values()
combined.corr()['In'].apply(abs).sort_values()


journeys_count_df['day'] = journeys_count_df['Time'].dt.dayofyear
daily_count_df = journeys_count_df.groupby(['day'])[['In', 'Out']].sum()
daily_data_df.index = daily_data_df['Time'].dt.dayofyear
daily_data_df.index.name = 'day'

daily_count_df = daily_count_df.reset_index()
daily_data_df = daily_data_df.reset_index()
daily_combined = daily_count_df.merge(daily_data_df)

daily_combined.corr()['Out'].apply(abs).sort_values()
daily_combined.corr()['In'].apply(abs).sort_values()

daily_data_df.columns
hourly_data_df.columns

daily_columns = ['Time', 'rain_d', 'clear_d', 'cloudy_d', 'apparentTemperatureHigh_d',
       'apparentTemperatureLow_d', 'precipIntensity_d', 'dewPoint_d',
       'humidity_d', 'windSpeed_d', 'uvIndex_d', 'visibility_d']


hourly_columns = ['Time', 'rain_d', 'clear_d', 'cloudy_d', 'apparentTemperature_d',
       'precipIntensity_d', 'dewPoint_d', 'humidity_d', 'windSpeed_d',
       'uvIndex_d', 'visibility_d']
