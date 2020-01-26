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

%matplotlib inline

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
# stations.describe()
# len(stations['Station ID'].unique())

# 2. Journey Data
journeys_df = pd.read_csv('data/london/journeys.csv')
# journeys_df.info()
# journeys_df.head()
# journeys_df.describe()

# Data Cleaning
fig = plt.figure(figsize=(10, 8))
plt.boxplot(journeys_df['Journey Duration'])
plt.title('Journey Duration Boxplot (Before Removing Outliers)', size=25, pad=20)
plt.xticks(fontsize=0)
plt.yticks(fontsize = 15)
fig.savefig('images/journey_duration_boxplot_before_cleaning', dpi = 200)

# drop rows which have less than 1 min duration
journeys_df = journeys_df[journeys_df['Journey Duration'] >= 60]

# drop rows with outlier journey duration
Q1 = journeys_df['Journey Duration'].quantile(0.25)
Q3 = journeys_df['Journey Duration'].quantile(0.75)
IQR = Q3 - Q1
journeys_df = journeys_df[(journeys_df['Journey Duration'] > (Q1 - 1.5 * IQR)) & (journeys_df['Journey Duration'] < (Q3 + 1.5 * IQR))]

fig = plt.figure(figsize=(10, 8))
plt.boxplot(journeys_df['Journey Duration'])
plt.title('Journey Duration Boxplot (After Removing Outliers)', size=25, pad=20)
plt.xticks(fontsize=0)
plt.yticks(fontsize = 15)
fig.savefig('images/journey_duration_boxplot_after_cleaning', dpi = 200)

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

journeys_df.info()
# journeys_df.head()
journeys_df.to_csv('data/processed/london_clean_journeys.csv', index=False)
# journeys_df.describe()
journeys_df['Start Time'].min() #Timestamp('2017-08-01 00:00:00')
journeys_df['Start Time'].max() #Timestamp('2017-09-19 23:54:00')
