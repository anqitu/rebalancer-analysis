import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime
import os
import re

# %matplotlib inlines
# 1. Station Data
stations = pd.read_csv('data/raw/stations.csv')
# stations.info()
# stations.head()
# stations.describe()
len(stations['Station ID'].unique())  ## 773

# 2. Journey Data
START_DATE = datetime(year = 2018, month = 1, day = 1, hour = 0)
END_DATE = datetime(year = 2019, month = 1, day = 1, hour = 0)

TFL_JOURNEYS_DATA_DIR = 'data/raw/journeys-tfl/2018'
filenames = sorted(os.listdir(TFL_JOURNEYS_DATA_DIR))
file_ids = [int(re.findall('\d+', filename)[0]) for filename in filenames]
file_count = len(file_ids) # 53
max(file_ids) - min(file_ids) + 1 # 53

def count_file_dates(file_id):
    file_order = file_id - min(file_ids) +1
    days = list(range((file_order-1)*7 - 4, (file_order-1)*7 + 3))
    for index, day in enumerate(days):
        if day <= 0:
            days[index] = 365 + day
    return days
count_file_dates(90)
journeys_df = []

filepath = os.path.join(TFL_JOURNEYS_DATA_DIR, '90JourneyDataExtract27Dec2017-02Jan2018.csv')
df = pd.read_csv(filepath)
id_name_df = df[['EndStation Id', 'EndStation Name']].drop_duplicates()

for filename in filenames:
    file_id = int(re.findall('\d+', filename)[0])
    print('Loading {} ({}/{})'.format(filename, file_id - min(file_ids) + 1, file_count))
    filepath = os.path.join(TFL_JOURNEYS_DATA_DIR, filename)

    if file_id == 134:
        df = pd.read_csv(filepath, usecols=['Duration', 'End Date', 'EndStation Name', 'Start Date', 'StartStation Id'])
        df = df.merge(id_name_df).drop(columns = ['EndStation Name'])
    else:
        df = pd.read_csv(filepath, usecols=['Duration', 'End Date', 'EndStation Id', 'Start Date', 'StartStation Id'])
    df['Start Date'] = pd.to_datetime(df['Start Date'], format = '%d/%m/%Y %H:%M')
    df['End Date'] = pd.to_datetime(df['End Date'], format = '%d/%m/%Y %H:%M')
    journeys_df.append(df)

    print(set(df['Start Date'].dt.dayofyear.unique()).difference(set(list(count_file_dates(file_id)))))
    print(df[(df['End Date'] - df['Start Date']).astype('timedelta64[m]').astype(int) * 60 != df['Duration']].shape[0])
    print(df[df['Start Date'] > df['End Date']].shape[0])
    print(df[df['Duration'] > 3600].shape[0] / df.shape[0])

# filepath = os.path.join(TFL_JOURNEYS_DATA_DIR, '134JourneyDataExtract31Oct2018-06Nov2018.csv')
# df = pd.read_csv(filepath, usecols=['Duration', 'End Date', 'EndStation Id', 'Start Date', 'StartStation Id'])
# df = pd.read_csv(filepath, usecols=['Duration', 'End Date', 'EndStation Id', 'Start Date', 'StartStation Id'])
# pd.to_datetime(df['Start Date'], format = '%d/%m/%Y %H:%M').dt.dayofyear.unique()
# df.sort_values(['Duration'], ascending=False)['Start Date']

journeys_df = pd.concat(journeys_df)
journeys_df.info()
journeys_df.head()
journeys_df.shape[0] # 10495504

# Remove data before and after 2018
journeys_df[(journeys_df['Start Date'] < START_DATE) | (journeys_df['End Date'] >= END_DATE)].shape[0] # 58963
journeys_df = journeys_df[(journeys_df['Start Date'] >= START_DATE) & (journeys_df['End Date'] < END_DATE)]
journeys_df.shape[0] # 10436541
print('Cleaning data from {} to {}'.format(journeys_df['Start Date'].min(), journeys_df['End Date'].max()))
# Cleaning data from 2018-01-01 00:00:00 to 2018-12-31 23:59:00

# Remove data with start dates before end dates
# journeys_df[journeys_df['Start Date'] > journeys_df['End Date']].shape[0] # 0
# journeys_df = journeys_df[~(journeys_df['Start Date'] > journeys_df['End Date'])]
# journeys_df.shape[0]

len(journeys_df['StartStation Id'].unique())  # 798
# journeys_df.describe()

# Data Cleaning
fig = plt.figure(figsize=(10, 8))
plt.boxplot(journeys_df['Duration'])
plt.title('Journey Duration Boxplot (Before Removing Outliers)', size=25, pad=20)
plt.xticks(fontsize=0)
plt.yticks(fontsize = 15)
fig.savefig('images/journey_duration_boxplot_before_cleaning', dpi = 200, bbox_inches = 'tight')

# # drop rows which have less than 1 min duration
# journeys_df[journeys_df['Duration'] < 60].shape[0]
# journeys_df = journeys_df[journeys_df['Duration'] >= 60]
# journeys_df.shape[0]

# drop rows with outlier journey duration
Q1 = journeys_df['Duration'].quantile(0.25)
Q3 = journeys_df['Duration'].quantile(0.75)
IQR = Q3 - Q1
Q1 - 3 * IQR # -1860.0
Q3 + 3 * IQR # 3600.0
journeys_df[~((journeys_df['Duration'] > (Q1 - 3 * IQR)) & (journeys_df['Duration'] < (Q3 + 3 * IQR)))].shape[0] # 364300
journeys_df = journeys_df[(journeys_df['Duration'] > (Q1 - 3 * IQR)) & (journeys_df['Duration'] < (Q3 + 3 * IQR))]
journeys_df.shape[0] # 10072241
364300/10436541 # 0.0349

fig = plt.figure(figsize=(10, 8))
plt.boxplot(journeys_df['Duration'])
plt.title('Journey Duration Boxplot (After Removing Outliers)', size=25, pad=20)
plt.xticks(fontsize=0)
plt.yticks(fontsize = 15)
fig.savefig('images/journey_duration_boxplot_after_cleaning', dpi = 200, bbox_inches = 'tight')

# journeys_df.info()
# journeys_df.head()
journeys_df.to_csv('data/processed/london_clean_journeys.csv', index=False)
# journeys_df.describe()
journeys_df['Start Date'].min() # Timestamp('2019-01-01 00:00:00')
journeys_df['Start Date'].max() # Timestamp('2019-12-31 23:56:00')

journeys_df['End Date'].min() # Timestamp('2019-01-01 00:08:00')
journeys_df['End Date'].max() # Timestamp('2019-12-31 23:59:00')
