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

stations = pd.read_csv('data/london/stations.csv')
journeys_df = pd.read_csv('data/processed/london_clean_journeys.csv', parse_dates=['Start Time', 'End Time'])
# journeys_df.info()
# journeys_df.head()
journeys_df.describe()

# group journeys by Station ID and by set window intervals
def get_time_groups(grouper_key, granularity, groupby):
    grouper = pd.Grouper(key=grouper_key, freq=str(granularity) + 'Min', label='right')
    groups = journeys_df.groupby([groupby, grouper]).size()
    groups = groups.unstack(fill_value=0).stack() # fill nonexistent counts as 0
    return groups.reset_index()

hours = 2
granularity = 60 * hours # minutes

journeys_count_df = get_time_groups('Start Time', granularity, 'Start Station ID')
journeys_count_df = journeys_count_df.rename(columns={0: 'Out', 'Start Station ID': 'Station ID', 'Start Time': 'Time'})
journeys_count_df['In'] = get_time_groups('End Time', granularity, 'End Station ID')[0]
journeys_count_df['Time'] = journeys_count_df['Time'] - pd.Timedelta(hours=hours)
journeys_count_df.head(10)
journeys_count_df.describe()

# journeys_df[(journeys_df['Start Time'] < datetime(2017, 8, 1, 8)) & (journeys_df['Start Station ID']==1)]

# Remove stations with no journeys
stations_no_journey = set(stations['Station ID'].unique()).difference(set(journeys_count_df['Station ID'].unique()))
stations = stations[~stations['Station ID'].isin(stations_no_journey)]
len(stations['Station ID'].unique())

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
journeys_count_df.to_csv('data/processed/london_journeys_count_with_{}h_interval.csv'.format(hours), index=False)


# Average Station Hourly Demand
journeys_count_df['Weekday'] = journeys_count_df['Time'].apply(lambda x: calendar.day_name[x.weekday()])
journeys_count_df['Hour'] = journeys_count_df['Time'].apply(lambda x: x.hour)
journeys_count_df['Week Hour'] = journeys_count_df['Time'].apply(lambda x: x.weekday() * 24 + x.hour)
weeks = [g for n, g in journeys_count_df.groupby(pd.Grouper(key='Time', freq='W'))]

fig = plt.figure(figsize=(15, 8))
plt.title('Average Station 2-hour Interval Demand by Week', size=25, pad=20)
colors = sns.cubehelix_palette(8)[1:]
for i, week in enumerate(weeks[1:-1]):
    sns.lineplot(week['Week Hour'], week['Out'], color=colors[i], ci=None, label="Week {}".format(i+1))
plt.xlabel('DayOfWeek', size=20)
plt.ylabel('Trips', size=20)
plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
plt.yticks(size=15)
plt.legend(fontsize=15)
fig.savefig('images/avg_station_hourly_demand', dpi = 200)

fig = plt.figure(figsize=(15, 8))
plt.title('Average Station 2-hour Interval Supply by Week', size=25, pad=20)
colors = sns.cubehelix_palette(8)[1:]
for i, week in enumerate(weeks[1:-1]):
    sns.lineplot(week['Week Hour'], week['In'], color=colors[i], ci=None, label="Week {}".format(i+1))
plt.xlabel('DayOfWeek', size=20)
plt.ylabel('Trips', size=20)
plt.xticks(size=15, ticks=range(15, 24*7, 24), labels=list(calendar.day_name))
plt.yticks(size=15)
plt.legend(fontsize=15)
fig.savefig('images/avg_station_hourly_supply', dpi = 200)
