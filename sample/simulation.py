import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

journeys_df = pd.read_csv('data/clean_journeys.csv', parse_dates=[14, 15])
clean_df = pd.read_csv('data/clean_pred.csv')
station_df = pd.read_csv('data/stations.csv')

# create time column
time_columns = ['Start Date', 'Start Month', 'Start Year', 'Start Hour',
'Start Minute', 'End Date', 'End Month', 'End Year', 'End Hour', 'End Minute']
journeys_df = journeys_df.drop(columns=time_columns)
supply_class_count = pd.crosstab(clean_df['Station ID'], clean_df['Supply'])

top_undersupplied_stations = supply_class_count.sort_values([1], ascending=False)[:10]
top_undersupplied_stations

station_id = 66
station_capacity = int(station_df[station_df['Station ID'] == station_id]['Capacity'])

station_journeys = []
for flow, prefix in zip(['In', 'Out'], ['Start', 'End']):
    uni_df = journeys_df[journeys_df[prefix + ' Station ID'] == station_id]
    uni_df = uni_df.rename(columns={prefix + ' Time': 'Time'})
    uni_df['Flow'] = flow
    station_journeys.append(uni_df)
station_journeys = pd.concat(station_journeys, sort=False)[['Time', 'Flow']].set_index('Time')
trips_by_hour = [(n.hour, g) for n, g in station_journeys.groupby(pd.Grouper(freq='H'))]
dist_ratios = np.arange(0, 1, 0.1)
results = pd.DataFrame(columns=['Capacity', 'Dist Ratio', 'Missed In', 'Missed Out'])
for capacity in range(0, 500, 50):
    for dist_ratio in dist_ratios:
        missed_out = 0
        missed_in = 0
        supply = round(capacity * dist_ratio)
        for hour, hour_trips in trips_by_hour:
            if hour in [5, 17]:
                supply = round(capacity * dist_ratio)
            for index, trip in hour_trips.iterrows():
                if trip['Flow'] == 'In':
                    if supply < capacity:
                        supply += 1
                    else:
                        missed_in += 1
                else:
                    if supply > 0:
                        supply -= 1
                    else:
                        missed_out += 1
        print(capacity, dist_ratio, missed_in, missed_out)
        results = results.append({'Capacity': capacity, 'Dist Ratio': dist_ratio, 'Missed In': missed_in, 'Missed Out': missed_out}, ignore_index=True)

results = results.astype(float)
fig = plt.figure(figsize=(24, 10))
for i, dist_ratio in enumerate(dist_ratios):
    plot_df = results[results['Dist Ratio'] == dist_ratio]
    plt.subplot(2, 5, i+1)
    sns.lineplot(plot_df['Capacity'], plot_df['Missed Out'], color='red', label='No Bikes')
    sns.lineplot(plot_df['Capacity'], plot_df['Missed In'], color='skyblue', label='No Space')
    plt.title('Distribution Ratio of {:.2f}'.format(dist_ratio))
    plt.xlabel('Capacity')
    plt.ylabel('Trips')
fig.savefig('images/capacity_optimised')
