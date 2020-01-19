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

from geopy.distance import distance
from datetime import datetime

df = pd.read_csv('supply_demand_gap.csv')
df[df['Cycle1 (Bef)'] > 0]['Cycle1 (Bef)'].sum()

df = pd.read_csv('cycle_results.csv')
df


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
