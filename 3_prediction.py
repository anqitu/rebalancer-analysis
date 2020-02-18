TEST = True

TIMESTEPS = 12
LAGS = 2
RNN_LAYER_UNITS = 128
DENSE_LAYER_UNITS = 128
# WEATHER = 'daily'
# WEATHER = 'hourly'
WEATHER = ''

EPOCHS = 2
MODEL_EXPERIMENT_TIMES = 2
WORKING_DIR = "."

DAILY_COLUMNS = ['Time',
                 'rain_d',
                 'clear_d',
                 'cloudy_d',
                 'apparentTemperatureHigh_d',
                 'apparentTemperatureLow_d',
                 'precipIntensity_d',
                 'dewPoint_d',
                 'humidity_d',
                 'windSpeed_d',
                 'uvIndex_d',
                 'visibility_d',
                 ]

HOURLY_COLUMNS = ['Time',
                  'rain_h',
                  'clear_h',
                  'cloudy_h',
                  'apparentTemperature_h',
                  'precipIntensity_h',
                  'dewPoint_h',
                  'humidity_h',
                  'windSpeed_h',
                  'uvIndex_h',
                  'visibility_h',
                  ]

import warnings
warnings.simplefilter("ignore")

"""Set SEED to get reproducible results"""
SEED = 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(SEED)

# 2. Set `python` built-in pseudo-random generator at a fixed value
from random import seed
seed(SEED)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(SEED)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
from tensorflow import random
random.set_seed(SEED)

# # 5. Configure a new global `tensorflow` session
# from tensorflow.keras import backend as K
# # session_conf = tf.config.experimental(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# # 6. Initialize weights
# from tensorflow import keras
# kernel_initializer = keras.initializers.glorot_uniform(seed=SEED)
# bias_initializer=keras.initializers.Constant(value=0.1)

import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


def current_time():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def print_info(info):
    print("{:<6} {}: {}".format('[INFO]', current_time(), info))

def get_rmse(y_test, y_pred):
    return mean_squared_error(y_test, y_pred) ** 0.5

TEST_PREDICT_START_TIME = datetime(year = 2018, month = 10, day = 1, hour = 0)

# journeys_count_df = pd.read_csv(WORKING_DIR + '/data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df.info()
# journeys_count_df.head(10)
# journeys_count_df.describe()
#      Out	    In
# std  5.575693 5.776871

# journeys_count_df = journeys_count_df[journeys_count_df['Station ID'] <= 2]

"""Same hour of Previous day (Same hour)"""
# journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
#
# predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].shift(1)
# journeys_count_df['In(Predict)'] = predict_in.round(0)
# predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].shift(1)
# journeys_count_df['Out(Predict)'] = predict_out.round(0)
#
# journeys_count_df[pd.isna(journeys_count_df['In(Predict)'])]
# # journeys_count_df.head(30)
# # journeys_count_df.iloc[599:650]
#
# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 4.276613034755501
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 4.251557661842484
#
# predict_df = journeys_count_df[journeys_count_df['Time'] >= TEST_PREDICT_START_TIME]
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 4.063323480712249
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 4.013350239527448
#
# journeys_count_df = journeys_count_df.drop(columns = ['In(Predict)', 'Out(Predict)'])

"""Same hour of past 7th date (Same hour of same dayofweek)"""
# journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
#
# predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].shift(7)
# journeys_count_df['In(Predict)'] = predict_in.round(0)
# predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].shift(7)
# journeys_count_df['Out(Predict)'] = predict_out.round(0)
#
# # journeys_count_df.head(90)
# # journeys_count_df[pd.isna(journeys_count_df['In(Predict)'])]
#
# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.6700835744865685
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.7639595094969187
#
# TRAIN_PREDICT_START_TIME = datetime(year = 2018, month = 1, day = 8, hour = 0)
#
# journeys_count_df = journeys_count_df[(journeys_count_df['Time'] >= TRAIN_PREDICT_START_TIME)]
# LWSDSH_all_RMSE = get_rmse(np.concatenate([journeys_count_df['In'], journeys_count_df['Out']]),
#     np.concatenate([journeys_count_df['In(Predict)'], journeys_count_df['Out(Predict)']]))
# LWSDSH_all_RMSE # 3.717317892839432
#
# train_predict_df = journeys_count_df[(journeys_count_df['Time'] < TEST_PREDICT_START_TIME) & (journeys_count_df['Time'] >= TRAIN_PREDICT_START_TIME)]
# train_predict_df.shape[0]
# # train_predict_df.isnull().sum()
# train_predict_df['Time'].min() # Timestamp('2018-01-08 00:00:00')
# train_predict_df.shape[0] / 773 # 3192
#
# get_rmse(train_predict_df['In'], train_predict_df['In(Predict)']) # 3.7741348440706726
# get_rmse(train_predict_df['Out'], train_predict_df['Out(Predict)']) # 3.87066731303867
# LWSDSH_train_RMSE = get_rmse(np.concatenate([train_predict_df['In'], train_predict_df['Out']]),
#     np.concatenate([train_predict_df['In(Predict)'], train_predict_df['Out(Predict)']]))
# LWSDSH_train_RMSE # 3.8227058001796554
#
# test_predict_df = journeys_count_df[journeys_count_df['Time'] >= TEST_PREDICT_START_TIME]
# test_predict_df['Time'].min() # Timestamp('2018-10-01 00:00:00')
# get_rmse(test_predict_df['In'], test_predict_df['In(Predict)']) # 3.351114117793388
# get_rmse(test_predict_df['Out'], test_predict_df['Out(Predict)']) # 3.4368473870325738
# LWSDSH_test_RMSE = get_rmse(np.concatenate([test_predict_df['In'], test_predict_df['Out']]),
#     np.concatenate([test_predict_df['In(Predict)'], test_predict_df['Out(Predict)']]))
# LWSDSH_test_RMSE # 3.394251448569094
# test_predict_df.shape[0] / 773 # 1104
#
# predict_df = test_predict_df
# predict_df = predict_df.drop(columns = ['Out', 'In', 'Hour'])
# predict_df = predict_df.rename(columns = {'In(Predict)': 'In', 'Out(Predict)': 'Out'})
# predict_df['In'] =  predict_df['In'].astype(int)
# predict_df['Out'] =  predict_df['Out'].astype(int)
# predict_df.to_csv(WORKING_DIR + '/data/processed/london_journeys_predict_with_2h_interval_LWSDSH.csv', index = False)


"""Past 7 day MA of same hour"""
# journeys_count_df = pd.read_csv(WORKING_DIR + '/data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df['Hour'] = journeys_count_df['Time'].dt.hour
#
# predict_in = journeys_count_df.groupby(['Station ID', 'Hour'])[['In']].rolling(window=7).mean().shift(1)
# predict_in.index = predict_in.index.get_level_values(2)
# journeys_count_df['In(Predict)'] = predict_in.round(0)
# predict_out = journeys_count_df.groupby(['Station ID', 'Hour'])[['Out']].rolling(window=7).mean().shift(1)
# predict_out.index = predict_out.index.get_level_values(2)
# journeys_count_df['Out(Predict)'] = predict_out.round(0)
#
# # journeys_count_df.sort_values(['Station ID', 'Time'])
# # journeys_count_df[(journeys_count_df['Station ID'] == 1) & (journeys_count_df['Time'].dt.hour == 2)].head(50)
# # predict_in.head(50)
#
# # journeys_count_df.isnull().sum()
# # journeys_count_df.head(100)
# # journeys_count_df.sort_values(by = ['Station ID', 'Hour']).head(20)
#
# # predict_df = journeys_count_df.dropna()
# # get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.9255441212077766
# # get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.871822155661291
#
# TRAIN_PREDICT_START_TIME = datetime(year = 2018, month = 1, day = 8, hour = 0)
#
# journeys_count_df = journeys_count_df[(journeys_count_df['Time'] >= TRAIN_PREDICT_START_TIME)]
# P7MA_all_RMSE = get_rmse(np.concatenate([journeys_count_df['In'], journeys_count_df['Out']]),
#     np.concatenate([journeys_count_df['In(Predict)'], journeys_count_df['Out(Predict)']]))
# P7MA_all_RMSE # 3.6574283858384686
#
# train_predict_df = journeys_count_df[(journeys_count_df['Time'] < TEST_PREDICT_START_TIME) & (journeys_count_df['Time'] >= TRAIN_PREDICT_START_TIME)]
# train_predict_df.shape[0]
# # train_predict_df.isnull().sum()
# train_predict_df['Time'].min() # Timestamp('2018-01-08 00:00:00')
# train_predict_df.shape[0] / 773 # 3192
#
# get_rmse(train_predict_df['In'], train_predict_df['In(Predict)']) # 3.768349813860415
# get_rmse(train_predict_df['Out'], train_predict_df['Out(Predict)']) # 3.7300952310817412
# P7MA_train_RMSE = get_rmse(np.concatenate([train_predict_df['In'], train_predict_df['Out']]),
#     np.concatenate([train_predict_df['In(Predict)'], train_predict_df['Out(Predict)']]))
# P7MA_train_RMSE # 3.749271312706015
#
# test_predict_df = journeys_count_df[journeys_count_df['Time'] >= TEST_PREDICT_START_TIME]
# test_predict_df['Time'].min() # Timestamp('2018-10-01 00:00:00')
# get_rmse(test_predict_df['In'], test_predict_df['In(Predict)']) # 3.3970872705330595
# get_rmse(test_predict_df['Out'], test_predict_df['Out(Predict)']) # 3.3585303454534854
# P7MA_test_RMSE = get_rmse(np.concatenate([test_predict_df['In'], test_predict_df['Out']]),
#     np.concatenate([test_predict_df['In(Predict)'], test_predict_df['Out(Predict)']]))
# P7MA_test_RMSE # 3.3778638223698167
# test_predict_df.shape[0] / 773 # 1104
#
# predict_df = test_predict_df
# predict_df = predict_df.drop(columns = ['Out', 'In', 'Hour'])
# predict_df = predict_df.rename(columns = {'In(Predict)': 'In', 'Out(Predict)': 'Out'})
# predict_df['In'] =  predict_df['In'].astype(int)
# predict_df['Out'] =  predict_df['Out'].astype(int)
#
# predict_df2 = predict_df.copy()
# predict_df['Lag'] = 0
#
# predict_df2['Time'] = predict_df2['Time'] - pd.Timedelta(hours=2)
# predict_df2['Lag'] = 1
#
# combine = predict_df.append(predict_df2)
# combine.to_csv(WORKING_DIR + '/data/predicted/london_journeys_predict_with_2h_interval_7DMA.csv', index = False)

P7MA_train_RMSE = 3.749271312706015
P7MA_test_RMSE = 3.3778638223698167

"""Modelling"""
def get_counts_df(journeys_count_df):
    print('Processing Counts Data')
    counts_df = []
    features = ['In', 'Out']
    for feature in features:
        feature_values_df = journeys_count_df.pivot(index='Time', columns='Station ID', values=feature)
        feature_values_df.columns = ['{}_{}'.format(feature, col) for col in feature_values_df.columns]
        counts_df.append(feature_values_df)
    counts_df = pd.concat(counts_df, axis = 1)
    return counts_df

def get_influencial_features_df(journeys_count_df):
    print('Processing Influential Features Data')
    features_df = journeys_count_df[['Time']]
    features_df = features_df.drop_duplicates()

    if WEATHER == 'hourly':
        print('Adding hourly weather info ----------')
        weather_df = pd.read_csv(WORKING_DIR + '/data/processed/london_hourly_weather.csv', parse_dates=['Time'])[HOURLY_COLUMNS]
        features_df = features_df.merge(weather_df)
    elif WEATHER == 'daily':
        print('Adding daily weather info ----------')
        daily_data_df = pd.read_csv(WORKING_DIR + '/data/processed/london_daily_weather.csv', parse_dates=['Time'])[DAILY_COLUMNS]
        features_df = features_df.merge(daily_data_df, how = 'left')
        features_df = features_df.fillna(method = 'ffill')
    elif WEATHER == 'both':
        print('Adding both daily and hourly weather info ----------')
        weather_df = pd.read_csv(WORKING_DIR + '/data/processed/london_hourly_weather.csv', parse_dates=['Time'])[HOURLY_COLUMNS]
        features_df = features_df.merge(weather_df)

        daily_data_df = pd.read_csv(WORKING_DIR + '/data/processed/london_daily_weather.csv', parse_dates=['Time'])[DAILY_COLUMNS]
        features_df = features_df.merge(daily_data_df, how = 'left')
        features_df = features_df.fillna(method = 'ffill')
    else:
        print('Adding no weather info ----------')

    features_df['Hour'] = features_df['Time'].dt.hour
    features_df['DayOfWeek'] = features_df['Time'].dt.dayofweek
    features_df = features_df.set_index('Time')
    return features_df

def get_features_df(journeys_count_df):
    counts_df = get_counts_df(journeys_count_df)
    influential_features_df = get_influencial_features_df(journeys_count_df)
    features_df = pd.concat([counts_df, influential_features_df], axis = 1)
    return features_df

def get_historic_features_df(features_df, timesteps = 1, lags = 1):
    print('Processing Historic Features Data')
    x = []
    for timestep in reversed(range(1, timesteps+1)):
        historic_features_df = features_df.shift(timestep)
        historic_features_df.columns = ['{}(t-{})'.format(col, timestep) for col in historic_features_df.columns]
        x.append(historic_features_df)

    for lag in range(lags):
        historic_features_df = features_df.shift(-lag)
        historic_features_df.columns = ['{}(t+{})'.format(col, lag) for col in historic_features_df.columns]
        x.append(historic_features_df)

    x = pd.concat(x, axis = 1)
    x = x.dropna()

    return x

def scale_x(x):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    return x

def reshape_inputs(x, timesteps):
    # reshape x to be 3D [samples, timesteps, features]
    feature_count = x.shape[1]//timesteps
    x = x.reshape((x.shape[0], timesteps, feature_count))
    print('x shape: {}'.format(x.shape))

    return x

def create_inputs(historic_df, timesteps = 1, lags = 1):
    inputs = historic_df.iloc[:, :historic_df.shape[1]*timesteps//(timesteps+lags)]
    inputs = scale_x(inputs)
    inputs = reshape_inputs(inputs, timesteps)
    return inputs

def create_outputs(historic_df, timesteps = 1, lags = 1):
    outputs = historic_df.iloc[:, -historic_df.shape[1]*lags//(timesteps+lags):]
    n_features = outputs.shape[1]//lags
    output_indexes = [i for i in range(len(outputs.columns)) if i%n_features in range(station_count*2)]
    outputs = outputs.iloc[:, output_indexes]

    print('y shape: {}'.format(outputs.shape))

    return outputs.astype(int)

def prepare_data(journeys_count_df, timesteps = 1, lags = 1):
    features_df = get_features_df(journeys_count_df)
    historic_df = get_historic_features_df(features_df, timesteps, lags)
    x = create_inputs(historic_df, timesteps, lags)
    y = create_outputs(historic_df, timesteps, lags)

    # split into train and test
    test_len = sum(historic_df.index >= TEST_PREDICT_START_TIME)
    train_X, train_y = x[:-test_len], y[:-test_len]
    test_X, test_y = x[-test_len:], y[-test_len:]

    print('train_X shape: {}'.format(train_X.shape))
    print('test_X shape: {}'.format(test_X.shape))
    print('train_y shape: {}'.format(train_y.shape))
    print('test_y shape: {}'.format(test_y.shape))

    return train_X, train_y, test_X, test_y

def train_model(model, train_X, train_y, epochs=200, verbose = 2):
    earlystopping = EarlyStopping(patience=10, monitor='val_loss',
                                  verbose=2, restore_best_weights=True)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=32,
                    validation_split=0.3, shuffle=False, verbose=verbose,
                    callbacks=[earlystopping])
    return history

def plot_training_history(model_name, model, history, test_X, test_y):
    rmse = model.evaluate(test_X, test_y, verbose = 0) ** 0.5
    print('RMSE (float) = {:.3f}'.format(rmse))

    title = '{} Training History'.format(model_name)
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.plot([], [], ' ', label='Val RMSE = {:.3f}'.format(rmse))
    plt.legend(fontsize = 20)
    plt.title(title, size = 25, pad=20)
    plt.xlabel('No. Epochs', fontsize = 20)
    plt.ylabel('MSE', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(os.path.join(WORKING_DIR + '/images', title), dpi = 200, bbox_inches = 'tight')

def make_prediction(model, test_y):
    # Round to integer and convert negative number to 0
    y_pred = model.predict(test_X, verbose = 0).round(0).astype(int).clip(0)
    predict_df = pd.DataFrame(data = y_pred, index = test_y.index, columns = test_y.columns)
    print('RMSE (int) = {:.3f}'.format(get_rmse(predict_df.values.flatten(), test_y.values.flatten())))

    return predict_df

def postprocess_prediction(model_name, predict_df):
    predict_df = predict_df.unstack().reset_index()
    predict_df['Station ID'] = predict_df['level_0'].str.split('_').str.get(1).str.split('(').str.get(0)
    predict_df['in_out'] = predict_df['level_0'].str.split('_').str.get(0)
    predict_df['Lag'] = predict_df['level_0'].str.split('+').str.get(1).str.split(')').str.get(0).astype(int)
    predict_df = predict_df.rename(columns = {0: 'count'})

    predict_in_df = predict_df[predict_df['in_out'] == 'In']
    predict_in_df = predict_in_df.rename(columns = {'count': 'In'})
    predict_in_df = predict_in_df[['Time', 'Station ID', 'In', 'Lag']]

    predict_out_df = predict_df[predict_df['in_out'] == 'Out']
    predict_out_df = predict_out_df.rename(columns = {'count': 'Out'})
    predict_out_df = predict_out_df[['Time', 'Station ID', 'Out', 'Lag']]

    predict_df = predict_in_df.merge(predict_out_df)

    filepath = WORKING_DIR + '/data/predicted/london_journeys_predict_with_2h_interval_{}.csv'.format(model_name)
    predict_df.to_csv(filepath, index = False)
    print('Save prediction to {}'.format(filepath))

"""LSTM"""
def build_lstm(x, y):
    model = Sequential()
    model.add(LSTM(RNN_LAYER_UNITS, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(DENSE_LAYER_UNITS, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y.shape[1], activation = 'linear'))
    model.compile(loss="mse", optimizer="adam")
    return model

"""GRU"""
def build_gru(x, y):
    model = Sequential()
    model.add(GRU(RNN_LAYER_UNITS, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(DENSE_LAYER_UNITS, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y.shape[1], activation = 'linear'))
    model.compile(loss="mse", optimizer="adam")
    return model

"""Bidirectional LSTM"""
def build_bi_lstm(x, y):
    model = Sequential()
    model.add(Bidirectional(LSTM(RNN_LAYER_UNITS),input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(DENSE_LAYER_UNITS, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y.shape[1], activation = 'linear'))
    model.compile(loss="mse", optimizer="adam")
    return model

journeys_count_df = pd.read_csv(WORKING_DIR + '/data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
if TEST:
    journeys_count_df = journeys_count_df[journeys_count_df['Station ID'] <= 2]
station_count = len(journeys_count_df['Station ID'].unique()) # 773
train_X, train_y, test_X, test_y = prepare_data(journeys_count_df, timesteps = TIMESTEPS, lags = LAGS)


"""Experiment Many Models"""
scores_df = pd.DataFrame(columns = ['data', 'model', 'RMSE'])
get_models = [build_lstm, build_gru, build_bi_lstm]
model_names = ['LSTM', 'GRU', 'Bi-LSTM']

result_dir = WORKING_DIR + '/results/{} {} {} {}'.format(TIMESTEPS, RNN_LAYER_UNITS, DENSE_LAYER_UNITS, WEATHER).strip()
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

for i in range(MODEL_EXPERIMENT_TIMES):
    print_info('Training {}/{} ---------------------------------'.format(i, MODEL_EXPERIMENT_TIMES))
    for get_model, name in zip(get_models, model_names):
        print_info('Training {}'.format(name))
        model = get_model(train_X, train_y)
        train_model(model, train_X, train_y.values, EPOCHS, verbose = 0)
        scores_df.loc[scores_df.shape[0]] = ['Train', name, model.evaluate(train_X, train_y.values, verbose = 0) ** 0.5]
        scores_df.loc[scores_df.shape[0]] = ['Test', name, model.evaluate(test_X, test_y.values, verbose = 0) ** 0.5]

        scores_df = scores_df.sort_values(['model', 'data'])
        scores_df.to_csv(result_dir + '/model_performence.csv', index = False)
        print(scores_df)

    # scores_df = pd.read_csv(result_dir + '/model_performence.csv')
    fig = sns.catplot(data = scores_df, x = 'model', y = 'RMSE', kind = 'box',
                      height = 6, aspect = 1.4, legend=False,
                      hue = 'data',
                      hue_order = ['Train', 'Test'],
                      order=['LSTM', 'GRU', 'Bi-LSTM'])
    plt.title('Scores Distribution Across Models', size = 25, pad=20)
    plt.xlabel('Models', fontsize = 20)
    plt.ylabel('RMSE', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    # plt.legend(fontsize = 20)
    plt.legend(fontsize = 20, bbox_to_anchor=(1.3, 0.65))
    fig.savefig(result_dir + '/scores_boxplot', dpi = 200, bbox_inches = 'tight')

    mean_scores_df = scores_df.groupby(['data', 'model'])['RMSE'].mean().reset_index()
    mean_scores_df.loc[mean_scores_df.shape[0]] = ['Train', '7DMA', P7MA_train_RMSE]
    mean_scores_df.loc[mean_scores_df.shape[0]] = ['Test', '7DMA', P7MA_test_RMSE]
    mean_scores_df = mean_scores_df.sort_values(['RMSE'])
    mean_scores_df.to_csv(result_dir + '/model_mean_RMSE.csv', index = False)

    print(mean_scores_df)

    # mean_scores_df = pd.read_csv(result_dir + '/model_mean_RMSE.csv')
    fig = sns.catplot(data = mean_scores_df, x = 'model', y = 'RMSE',kind = 'bar',
                      height = 6, aspect = 1.4, legend=False,
                      order=['7DMA', 'LSTM', 'GRU', 'Bi-LSTM'],
                      hue = 'data', hue_order=['Train', 'Test'])
    plt.title('Mean RMSEs Across Models', size = 25, pad=20)
    plt.xlabel('Models', fontsize = 20)
    plt.ylabel('Mean RMSE', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(fontsize = 20)
    # plt.legend(fontsize = 20, bbox_to_anchor=(1, 0.65))
    for p in fig.ax.patches:
        fig.ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+0.05),
                        ha='center', va='bottom', color= 'black', fontsize=15)
    fig.savefig(result_dir + '/scores_mean', dpi = 200, bbox_inches = 'tight')


scores_df.describe()
scores_df = scores_df.sort_values(['model', 'data'])
scores_df.to_csv(WORKING_DIR + '/results/model_performence.csv', index = False)

scores_df = pd.read_csv(WORKING_DIR + '/results/model_performence.csv')
fig = sns.catplot(data = scores_df, x = 'model', y = 'RMSE', kind = 'box',
                  height = 6, aspect = 1.4, legend=False,
                  hue = 'data',
                  hue_order = ['Train', 'Test'],
                  order=['LSTM', 'GRU', 'Bi-LSTM'])
plt.title('Scores Distribution Across Models', size = 25, pad=20)
plt.xlabel('Models', fontsize = 20)
plt.ylabel('RMSE', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
# plt.legend(fontsize = 20, bbox_to_anchor=(1.3, 0.65))
fig.savefig(WORKING_DIR + '/results/scores_boxplot', dpi = 200, bbox_inches = 'tight')

mean_scores_df = scores_df.groupby(['data', 'model'])['RMSE'].mean().reset_index()
mean_scores_df.loc[mean_scores_df.shape[0]] = ['Train', '7DMA', P7MA_train_RMSE]
mean_scores_df.loc[mean_scores_df.shape[0]] = ['Test', '7DMA', P7MA_test_RMSE]
mean_scores_df = mean_scores_df.sort_values(['RMSE'])
mean_scores_df.to_csv(WORKING_DIR + '/results/model_mean_RMSE.csv', index = False)
mean_scores_df

mean_scores_df = pd.read_csv(WORKING_DIR + '/results/model_mean_RMSE.csv')
fig = sns.catplot(data = mean_scores_df, x = 'model', y = 'RMSE',kind = 'bar',
                  height = 6, aspect = 1.4, legend=False,
                  order=['7DMA', 'LSTM', 'GRU', 'Bi-LSTM'],
                  hue = 'data', hue_order=['Train', 'Test'])
plt.title('Mean RMSEs Across Models', size = 25, pad=20)
plt.xlabel('Models', fontsize = 20)
plt.ylabel('Mean RMSE', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
# plt.legend(fontsize = 20, bbox_to_anchor=(1, 0.65))
for p in fig.ax.patches:
    fig.ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+0.05),
                    ha='center', va='bottom', color= 'black', fontsize=15)
fig.savefig(WORKING_DIR + '/results/scores_mean', dpi = 200, bbox_inches = 'tight')

"""Build Single Model"""
lstm = build_lstm(train_X, train_y)
history = train_model(lstm, train_X, train_y.values)
plot_training_history('LSTM', lstm, history, test_X, test_y.values)

lstm_predict_df = make_prediction(lstm, test_y)
postprocess_prediction('LSTM', lstm_predict_df)
save_model(lstm, WORKING_DIR + '/results/LSTM.h5')

plot_model(lstm, to_file=WORKING_DIR + '/images/LSTM.png', show_shapes=True, show_layer_names=False, dpi=200)


gru = build_gru(train_X, train_y)
history = train_model(gru, train_X, train_y.values)
plot_training_history('GRU', gru, history, test_X, test_y.values)

gru_predict_df = make_prediction(gru, test_y)
postprocess_prediction('GRU', gru_predict_df)
save_model(gru, WORKING_DIR + '/results/GRU.h5')

plot_model(gru, to_file=WORKING_DIR + '/images/GRU.png', show_shapes=True, show_layer_names=False, dpi=200)


bi_lstm = build_bi_lstm(train_X, train_y)
history = train_model(bi_lstm, train_X, train_y.values)
plot_training_history('Bi-LSTM', bi_lstm, history, test_X, test_y.values)

bi_lstm_predict_df = make_prediction(bi_lstm, test_y)
postprocess_prediction('Bi-LSTM', bi_lstm_predict_df)
save_model(bi_lstm, WORKING_DIR + '/results/Bi-LSTM.h5'

plot_model(bi_lstm, to_file=WORKING_DIR + '/images/Bi-LSTM.png', show_shapes=True, show_layer_names=False, dpi=200)

"""Make prediction given a time"""
journeys_predict_df = pd.read_csv(WORKING_DIR + '/data/predicted/london_journeys_predict_with_2h_interval_LSTM.csv', parse_dates=['Time'])
journeys_predict_df.tail()
time = TEST_PREDICT_START_TIME
records = journeys_predict_df[(journeys_predict_df['Time'] == time)]
records_cur = {row['Station ID']: {'in': int(row['In']), 'out': int(row['Out'])} for index, row in records[records['Lag'] == 0].iterrows()}
records_next = {row['Station ID']: {'in': int(row['In']), 'out': int(row['Out'])} for index, row in records[records['Lag'] == 1].iterrows()}

# predict_df = pd.read_csv(WORKING_DIR + '/data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# predict_df = predict_df[predict_df['Time']>=TEST_PREDICT_START_TIME]
# predict_df2 = predict_df.copy()
# predict_df['Lag'] = 0
#
# predict_df2['Time'] = predict_df2['Time'] - pd.Timedelta(hours=2)
# predict_df2['Lag'] = 1
#
# combine = predict_df.append(predict_df2)
# combine.to_csv(WORKING_DIR + '/data/predicted/london_journeys_predict_with_2h_interval_actual.csv', index = False)
