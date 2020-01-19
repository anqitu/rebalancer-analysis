from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

PREDICT_START_TIME = datetime(year = 2017, month = 9, day = 10, hour = 0)

def get_rmse(y_test, y_pred):
    return mean_squared_error(y_test, y_pred) ** 0.5

journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df.info()
# journeys_count_df.head(10)
# journeys_count_df.describe()
#      Out	    In
# std  5.705519	5.939836

# journeys_count_df = journeys_count_df[journeys_count_df['Station ID'] <= 2]

# """Same hour of Previous day (Same hour)"""
# journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
#
# predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].shift(1)
# journeys_count_df['In(Predict)'] = predict_in.round(0)
# predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].shift(1)
# journeys_count_df['Out(Predict)'] = predict_out.round(0)
#
# # journeys_count_df[pd.isna(journeys_count_df['In(Predict)'])]
# # journeys_count_df.head(30)
# # journeys_count_df.iloc[599:650]
#
# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 4.5472509019451115
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 4.527446168898598
#
# predict_df = journeys_count_df[journeys_count_df['Time'] >= PREDICT_START_TIME]
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 4.4038030492188005
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 4.362692858828626
#
# journeys_count_df = journeys_count_df.drop(columns = ['In(Predict)', 'Out(Predict)'])
#
# """Same hour of past 7th date (Same hour of same dayofweek)"""
# journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
#
# predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].shift(7)
# journeys_count_df['In(Predict)'] = predict_in.round(0)
# predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].shift(7)
# journeys_count_df['Out(Predict)'] = predict_out.round(0)
#
# # journeys_count_df.head(90)
#
# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 4.0487652838133315
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 4.168556656066862
#
# predict_df = journeys_count_df[journeys_count_df['Time'] >= PREDICT_START_TIME]
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.872670782859088
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.966964623457324
#
# journeys_count_df = journeys_count_df.drop(columns = ['In(Predict)', 'Out(Predict)'])
#
"""Past 7 day MA of same hour"""
journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour

predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].rolling(window=7).mean().shift(1)
predict_in.index = predict_in.index.get_level_values(2)
journeys_count_df['In(Predict)'] = predict_in.round(0)
predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].rolling(window=7).mean().shift(1)
predict_out.index = predict_out.index.get_level_values(2)
journeys_count_df['Out(Predict)'] = predict_out.round(0)

# journeys_count_df.isnull().sum()
# journeys_count_df.head(100)
# journeys_count_df.sort_values(by = ['Station ID', 'Hour']).head(20)

# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.9255441212077766
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.871822155661291

predict_df = journeys_count_df[journeys_count_df['Time'] >= PREDICT_START_TIME]
get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.619754257055577
get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.5542678366316647

predict_df = predict_df.drop(columns = ['Out', 'In', 'Hour'])
predict_df = predict_df.rename(columns = {'In(Predict)': 'In', 'Out(Predict)': 'Out'})
predict_df['In'] =  predict_df['In'].astype(int)
predict_df['Out'] =  predict_df['Out'].astype(int)

predict_df.to_csv('data/processed/london_journeys_predict_with_2h_interval_7DMA.csv', index = False)

"""LSTM"""
journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])

# Preprocess
journeys_count_df['DayOfWeek'] = journeys_count_df['Time'].dt.dayofweek
journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
journeys_count_df.head()

features = ['In', 'Out', 'Hour', 'DayOfWeek']
features_df = []
for feature in features:
    feature_values_df = journeys_count_df.pivot(index='Time', columns='Station ID', values=feature)
    feature_values_df.columns = ['{}_{}'.format(feature, col) for col in feature_values_df.columns]
    features_df.append(feature_values_df)

features_df = pd.concat(features_df, axis = 1)
# features_df.head(10)

x = []
timesteps = 4
for periods in range(1, timesteps+1):
    historic_features_df = features_df.shift(periods)
    historic_features_df.columns = ['{}(t-{})'.format(col, periods) for col in historic_features_df.columns]
    x.append(historic_features_df)

x.reverse()
x = pd.concat(x, axis = 1)
x = x.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)

# reshape input to be 3D [samples, timesteps, features]
features = x.shape[1]//timesteps
station_count = len(journeys_count_df['Station ID'].unique()) # 779
x = x.reshape((x.shape[0], timesteps, features))
y = features_df.values[timesteps:, :station_count*2]
x.shape
y.shape


# split into input and outputs
test_len = sum(features_df.index >= PREDICT_START_TIME)
train_X, train_y = x[:-test_len], y[:-test_len]
test_X, test_y = x[-test_len:], y[-test_len:]

def build_model():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(x.shape[1], x.shape[2]),
                   activation = 'relu'))
    model.add(Dense(100))
    model.add(Dense(station_count*2))
    model.compile(loss="mse", optimizer="adam")
    return model

nn = build_model()
earlystopping = EarlyStopping(patience=10, monitor='val_loss',
                              verbose=2, restore_best_weights=True)
history = nn.fit(train_X, train_y, epochs=200, batch_size=32,
                validation_data=(test_X, test_y), verbose=2, shuffle=False,
                callbacks=[earlystopping])

# plt.figure(figsize=(12, 8))
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

nn.evaluate(test_X, test_y) ** 0.5

# Round to integer and convert negative number to 0
y_pred = nn.predict(test_X).round(0).astype(int).clip(0)
predict_df = pd.DataFrame(data = y_pred, index = features_df.index[features_df.index >= PREDICT_START_TIME], columns = features_df.columns[:station_count*2])

# get_rmse(predict_df.values.flatten(), test_y.flatten())
# predict_df.info()

# Postprocess prediction results
predict_df = predict_df.unstack().reset_index()
predict_df['Station ID'] = predict_df['level_0'].str.split('_').str.get(1)
predict_df['in_out'] = predict_df['level_0'].str.split('_').str.get(0)
predict_df = predict_df.rename(columns = {0: 'count'})

predict_in_df = predict_df[predict_df['in_out'] == 'In']
predict_in_df = predict_in_df.rename(columns = {'count': 'In'})
predict_in_df = predict_in_df[['Time', 'Station ID', 'In']]

predict_out_df = predict_df[predict_df['in_out'] == 'Out']
predict_out_df = predict_out_df.rename(columns = {'count': 'Out'})
predict_out_df = predict_out_df[['Time', 'Station ID', 'Out']]

predict_df = predict_in_df.merge(predict_out_df)

predict_df.to_csv('data/processed/london_journeys_predict_with_2h_interval_lstm.csv', index = False)

# Make prediction given a time
journeys_predict_df = pd.read_csv('data/processed/london_journeys_predict_with_2h_interval.csv', parse_dates=['Time'])
time = PREDICT_START_TIME
records = journeys_predict_df[(journeys_predict_df['Time'] == time)]
records = {row['Station ID']: {'in': int(row['In']), 'out': int(row['Out'])} for index, row in records.iterrows()}
