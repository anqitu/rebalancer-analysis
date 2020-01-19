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
