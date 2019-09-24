import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime
import calendar
from scipy import stats
import json

SEED = 2019
np.random.seed(SEED)
sns.set()

from matplotlib import rcParams
rcParams['figure.figsize'] = (20.0, 10.0)

### DATA EXPLORATION

# 1. Station Data
stations = pd.read_csv('data/london/stations.csv')
# stations.info()
# stations.head()
stations.describe()

# # Clustering
# Lat_Lng = stations[['Latitude', 'Longitude']]
# kmeans = KMeans(n_clusters=20, random_state=0).fit(Lat_Lng)
# stations['Cluster ID'] = kmeans.labels_
# stations['Cluster Center Latitude'] = [kmeans.cluster_centers_[id][0] for id in stations['Cluster ID']]
# stations['Cluster Center Longitude'] = [kmeans.cluster_centers_[id][1] for id in stations['Cluster ID']]
# clusters = stations.drop_duplicates(subset=['Cluster ID'])
# clusters = clusters.sort_values(['Cluster ID'])
# clusters['Cluster Capacity'] = list(stations.groupby(['Cluster ID'])['Capacity'].sum())
# sns.scatterplot(stations['Latitude'], stations['Longitude'], hue = stations['Cluster ID'])

# # Export to json
# stations_dict = []
# for index, row in clusters.iterrows():
#     station = {}
#     station['name'] = row['Station Name']
#     station['id'] = row['Cluster ID']
#     station['capacity'] = row['Cluster Capacity']
#     station['coordinates'] = [row['Cluster Center Longitude'], row['Cluster Center Latitude']]
#     stations_dict.append(station)
#
# with open('data/processed/london_clusters.json', 'w', encoding='utf-8') as f:
#     json.dump(stations_dict, f, ensure_ascii=False, indent=4)

# stations[(stations['Latitude'] < 51.47) & (stations['Longitude'] < -0.19)].shape
# stations = stations[(stations['Latitude'] < 51.47) & (stations['Longitude'] < -0.19)]

# 2. Journey Data
journeys_df = pd.read_csv('data/london/journeys.csv')
# journeys_df.info()
# journeys_df.head()
# journeys_df.describe()

# Data Cleaning
# drop rows which have zero duration
is_zero_duration = journeys_df['Journey Duration'] == 0
journeys_df[is_zero_duration].shape[0] # 1609
journeys_df = journeys_df[~is_zero_duration]

# drop rows with outlier journey duration
# journeys_df[['Journey Duration']].boxplot()
journeys_df[np.abs(stats.zscore(journeys_df['Journey Duration'])) >= 0.5].shape[0] #45247
journeys_df = journeys_df[np.abs(stats.zscore(journeys_df['Journey Duration'])) < 0.5]


# # Cluster
# journeys_df = journeys_df.merge(stations[['Station ID', 'Cluster ID']], left_on = 'Start Station ID', right_on = 'Station ID', how = 'left').rename(columns = {'Cluster ID': 'Start Cluster ID'})
# journeys_df = journeys_df.merge(stations[['Station ID', 'Cluster ID']], left_on = 'End Station ID', right_on = 'Station ID', how = 'left').rename(columns = {'Cluster ID': 'End Cluster ID'})
# journeys_df = journeys_df.dropna()
# journeys_df[journeys_df['Start Cluster ID'] == journeys_df['End Cluster ID']].shape
# journeys_df.shape
# journeys_df = journeys_df[journeys_df['Start Cluster ID'] != journeys_df['End Cluster ID']]

# reduce time columns into a single datetime column
time_columns = ['Start Date', 'Start Month', 'Start Year', 'Start Hour','Start Minute',
                'End Date', 'End Month', 'End Year', 'End Hour', 'End Minute']
journeys_df[time_columns] = journeys_df[time_columns].astype(str)
pad_zero = lambda x: '0' + x if len(x) == 1 else x
for prefix in ['Start', 'End']:
    date = journeys_df[prefix + ' Date'] + '/' + journeys_df[prefix + ' Month'] + '/' + journeys_df[prefix + ' Year']
    time = journeys_df[prefix + ' Hour'].apply(pad_zero) + journeys_df[prefix + ' Minute'].apply(pad_zero)
    time_str = date + ' ' + time
    journeys_df[prefix + ' Time'] = pd.to_datetime(time_str, format='%d/%m/%y %H%M')

# drop rows which have the same start and end time (invalid)
is_same_start_end_time = journeys_df['End Time'] == journeys_df['Start Time']
journeys_df[is_same_start_end_time].shape[0] # 2638
journeys_df = journeys_df[~is_same_start_end_time]

journeys_df.info()
journeys_df.head()
journeys_df.to_csv('data/processed/london_clean_journeys.csv', index=False)
journeys_df.describe()
journeys_df['Start Time'].min() #Timestamp('2017-08-01 00:00:00')
journeys_df['Start Time'].max() #Timestamp('2017-09-19 23:54:00')


# group journeys by Station ID and by set window intervals
def get_time_groups(grouper_key, granularity, groupby):
    grouper = pd.Grouper(key=grouper_key, freq=str(granularity) + 'Min', label='right')
    groups = journeys_df.groupby([groupby, grouper]).size()
    groups = groups.unstack(fill_value=0).stack() # fill nonexistent counts as 0
    return groups.reset_index()

hours = 1
granularity = 60 * hours # minutes

journeys_count_df = get_time_groups('Start Time', granularity, 'Start Station ID')
journeys_count_df = journeys_count_df.rename(columns={0: 'Out', 'Start Station ID': 'Station ID', 'Start Time': 'Time'})
journeys_count_df['In'] = get_time_groups('End Time', granularity, 'End Station ID')[0]
journeys_count_df['Time'] = journeys_count_df['Time'] - pd.Timedelta(hours=hours)
journeys_count_df.head(10)
journeys_count_df.describe()
for i, j in journeys_count_df.iterrows():
    break

# Remove stations with no journeys
stations_no_journey = set(stations['Station ID'].unique()).difference(set(journeys_count_df['Station ID'].unique()))
stations = stations[~stations['Station ID'].isin(stations_no_journey)]
# Export to json
stations_dict = []
for index, row in stations.iterrows():
    station = {}
    station['name'] = row['Station Name']
    station['id'] = row['Station ID']
    station['capacity'] = row['Capacity']
    station['coordinates'] = [row['Longitude'], row['Latitude']]
    stations_dict.append(station)
with open('data/processed/london_stations.json', 'w', encoding='utf-8') as f:
    json.dump(stations_dict, f, ensure_ascii=False, indent=4)

journeys_count_df = journeys_count_df[journeys_count_df['Station ID'].isin(stations['Station ID'])]
journeys_count_df.to_csv('data/processed/london_journeys_count_with_1h_interval.csv', index=False)


interval_hour = 6
grouper = pd.Grouper(key='Time', freq=str(interval_hour*60) + 'Min', label='right')
groups = journeys_count_df.groupby(['Station ID', grouper]).agg({'Out': 'sum', 'In': 'sum'})
groups = groups.reset_index()
groups['Delta'] = groups['In'] - groups['Out']
groups['Time'] = groups['Time'] - pd.Timedelta(hours=interval_hour)
groups.head(30)

import json
with open('result.json') as json_file:
    data = json.load(json_file)

data_df = {value['name']:[] for value in data[0]}
for record in data:
    for value in record:
        data_df[value['name']].append(value['value'])

data_df = pd.DataFrame(data_df)
data_df = data_df[[value['name'] for value in data[0]]]
data_df.to_csv('result.csv', index = False)

df = pd.DataFrame(columns = ['a', 'b'])
df = df.append({'a':1, 'b':2}, ignore_index=True)

df = pd.DataFrame.from_dict({'row_1': 3, 'row_2': 1}, orient='index')


from geopy.distance import distance
from datetime import datetime

df = pd.read_csv('supply_demand_gap.csv')
df[df['Cycle1 (Bef)'] > 0]['Cycle1 (Bef)'].sum()

df = pd.read_csv('cycle_results.csv')
df
# journeys_count_df = get_time_groups('Start', granularity, 'Cluster ID')
# journeys_count_df = journeys_count_df.rename(columns={0: 'Out', 'Start Cluster ID': 'Cluster ID', 'Start Time': 'Time'})
# journeys_count_df['In'] = get_time_groups('End', granularity, 'Cluster ID')[0]
# journeys_count_df['Delta'] = journeys_count_df['In'] - journeys_count_df['Out']
# journeys_count_df['Time'] = journeys_count_df['Time'] - pd.Timedelta(hours=hours)
# journeys_count_df.head(10)
# journeys_count_df.info()
# journeys_count_df.describe()
# journeys_count_df['Cluster ID'] = journeys_count_df['Cluster ID'].astype(int)
# journeys_count_df = journeys_count_df.rename(columns = {'Cluster ID': 'Station ID'})
# journeys_count_df.to_csv('data/processed/london_journeys_count_with_2h_interval_by_cluster.csv', index=False)
# journeys_df.sort_values(['Start Station ID', 'Start Time'])


# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
# journeys_count_df.sort_values(['Station ID', 'Hour']).head(20)
# ma = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].rolling(window=7).mean()
# ma.index = ma.index.get_level_values(2)
# journeys_count_df['MA'] = ma.round(0)

# 3. Visualization

# # Average Station Hourly Demand
# hourly_df = get_time_groups('Start', granularity, 'Cluster ID')
# hourly_df = hourly_df.rename(columns={0: 'Out', 'Start Station ID': 'Station ID', 'Start Time': 'Time'})
# hourly_df['Weekday'] = hourly_df['Time'].apply(lambda x: calendar.day_name[x.weekday()])
# hourly_df['Hour'] = hourly_df['Time'].apply(lambda x: x.hour)
# hourly_df['Week Hour'] = hourly_df['Time'].apply(lambda x: x.weekday() * 24 + x.hour)
#
# weeks = [g for n, g in hourly_df.groupby(pd.Grouper(key='Time', freq='W'))]
# fig = plt.figure(figsize=(24, 10))
# plt.title('Average Station Hourly Demand by Week', size=30)
# colors = sns.cubehelix_palette(8)[1:]
# for i, week in enumerate(weeks[1:-1]):
#     sns.lineplot(week['Week Hour'], week['Out'], color=colors[i], ci=None, label="Week {}".format(i+1))
# plt.xlabel('Weekday', size=24)
# plt.ylabel('Trips', size=24)
# plt.xticks(size=20, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
# plt.yticks(size=20)
# plt.legend(fontsize=24)
# fig.savefig('images/avg_station_hourly_demand')
#
# # station 1 incoming and outgoing trips
# station_1_sept_journeys = journeys_count_df[(journeys_count_df['Station ID'] == 1) & (journeys_count_df['Time'].apply(lambda t: t.month) == 8)]
# fig = plt.figure(figsize=(24, 10))
# plt.title('[Station 1] Number of Incoming and Outgoing Trips for September', size=30)
# sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['Out'], color='red', label='Trip End')
# sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['In'], color='skyblue', label='Trip Start')
# plt.xlabel('Time', size=24)
# plt.ylabel('Trips', size=24)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.legend(fontsize=24)
# fig.savefig('images/station_1_sept_journeys')
