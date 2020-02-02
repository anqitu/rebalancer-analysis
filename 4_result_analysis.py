import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv('data/results/cycle_results.csv', parse_dates=['Time'])
results_df['Hour'] = results_df['Time'].dt.hour
cycles = results_df.shape[0] * 2

fig = plt.figure(figsize=(12, 8))
plt.title('Demand-Supply Gap', size=25, pad=20)
plt.plot(results_df['Hour'], results_df['Supply Demand Gap Before Rebalance'], marker = 'o', markersize = 12)
plt.plot(results_df['Hour'], results_df['Supply Demand Gap After Rebalance'], marker = 'v', markersize = 12)
plt.xlabel('Hours in a Day', size=20)
plt.ylabel('No. of Shareable Bikes', size=20)
plt.xticks(size=15, ticks=range(0, cycles, 2))
plt.yticks(size=15)
plt.legend(fontsize=14)
fig.savefig('images/Demand Supply Gap', dpi = 200, bbox_inches = 'tight')

fig = plt.figure(figsize=(12, 8))
plt.title('Usage vs. Rebalance', size=25, pad=20)
plt.plot(results_df['Hour'], results_df['Moved Bikes'], marker = 'o', markersize = 12)
plt.plot(results_df['Hour'], results_df['Rebalanced Bikes'], marker = 'v', markersize = 12)
plt.xlabel('Hours in a Day', size=20)
plt.ylabel('No. of Shareable Bikes', size=20)
plt.xticks(size=15, ticks=range(0, cycles, 2))
plt.yticks(size=15)
plt.legend(fontsize=14)
fig.savefig('images/Usage vs Rebalance', dpi = 200, bbox_inches = 'tight')

# supply_demand_gap_df = pd.read_csv('data/results/supply_demand_gap.csv')
# supply_demand_gap_df[supply_demand_gap_df['Cycle8 (Bef)'] > 0]['Cycle8 (Bef)'].sum()
# supply_demand_gap_df[supply_demand_gap_df['Cycle8 (Aft)'] > 0]['Cycle8 (Aft)'].sum()
# supply_demand_gap_df.describe()

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
