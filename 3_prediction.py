import warnings
warnings.simplefilter("ignore")

"""Set SEED to get reproducible results"""
SEED = 2020

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(SEED)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(SEED)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(SEED)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(SEED)

# # 5. Configure a new global `tensorflow` session
# from tensorflow.keras import backend as K
# # session_conf = tf.config.experimental(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# # 6. Initialize weights
# from tensorflow import keras
# kernel_initializer = keras.initializers.glorot_uniform(seed=SEED)
# bias_initializer=keras.initializers.Constant(value=0.1)

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping

TEST_PREDICT_START_TIME = datetime(year = 2019, month = 9, day = 1, hour = 0)

def get_rmse(y_test, y_pred):
    return mean_squared_error(y_test, y_pred) ** 0.5

journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
# journeys_count_df.info()
# journeys_count_df.head(10)
journeys_count_df.describe()
#      Out	    In
# std  5.398425 5.599227

journeys_count_df = journeys_count_df[journeys_count_df['Station ID'] <= 2]

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

# journeys_count_df.sort_values(['Station ID', 'Time'])
# journeys_count_df[(journeys_count_df['Station ID'] == 1) & (journeys_count_df['Time'].dt.hour == 2)].head(50)
# predict_in.head(50)

# journeys_count_df.isnull().sum()
# journeys_count_df.head(100)
# journeys_count_df.sort_values(by = ['Station ID', 'Hour']).head(20)

# predict_df = journeys_count_df.dropna()
# get_rmse(predict_df['In'], predict_df['In(Predict)']) # 3.9255441212077766
# get_rmse(predict_df['Out'], predict_df['Out(Predict)']) # 3.871822155661291

TRAIN_PREDICT_START_TIME = datetime(year = 2019, month = 1, day = 8, hour = 0)
train_predict_df = journeys_count_df[(journeys_count_df['Time'] < TEST_PREDICT_START_TIME) & (journeys_count_df['Time'] >= TRAIN_PREDICT_START_TIME)]
# train_predict_df.shape
# train_predict_df.isnull().sum()
train_predict_df['Time'].min() # Timestamp('2019-01-08 00:00:00')
train_predict_df.shape[0] / 771 # 2832

get_rmse(train_predict_df['In'], train_predict_df['In(Predict)']) # 5.50112540671441
get_rmse(train_predict_df['Out'], train_predict_df['Out(Predict)']) # 3.666249708995289
P7MA_train_RMSE = get_rmse(np.concatenate([train_predict_df['In'], train_predict_df['Out']]),
    np.concatenate([train_predict_df['In(Predict)'], train_predict_df['Out(Predict)']]))
# 4.674599858228874

test_predict_df = journeys_count_df[journeys_count_df['Time'] >= TEST_PREDICT_START_TIME]
test_predict_df['Time'].min()
get_rmse(test_predict_df['In'], test_predict_df['In(Predict)']) # 4.523017029928338
get_rmse(test_predict_df['Out'], test_predict_df['Out(Predict)']) # 3.269560289610032
P7MA_test_RMSE = get_rmse(np.concatenate([test_predict_df['In'], test_predict_df['Out']]),
    np.concatenate([test_predict_df['In(Predict)'], test_predict_df['Out(Predict)']]))
# 3.94637222904889
test_predict_df.shape[0] / 771 # 1464

predict_df = test_predict_df
predict_df = predict_df.drop(columns = ['Out', 'In', 'Hour'])
predict_df = predict_df.rename(columns = {'In(Predict)': 'In', 'Out(Predict)': 'Out'})
predict_df['In'] =  predict_df['In'].astype(int)
predict_df['Out'] =  predict_df['Out'].astype(int)
predict_df.to_csv('data/processed/london_journeys_predict_with_2h_interval_7DMA.csv', index = False)

"""Modelling"""
def get_counts_df(journeys_count_df):
    features_df = []
    for feature in features[:2]:
        feature_values_df = journeys_count_df.pivot(index='Time', columns='Station ID', values=feature)
        feature_values_df.columns = ['{}_{}'.format(feature, col) for col in feature_values_df.columns]
        features_df.append(feature_values_df)
    features_df = pd.concat(features_df, axis = 1)
    return features_df

def get_influencial_features_df(journeys_count_df):
    features_df = journeys_count_df[['Time']]
    features_df = features_df.drop_duplicates()
    features_df['Hour'] = features_df['Time'].dt.hour
    features_df['DayOfWeek'] = features_df['Time'].dt.dayofweek
    features_df = features_df.set_index('Time')
    return features_df

def get_features_df(journeys_count_df):
    counts_df = get_counts_df(journeys_count_df)
    influential_features_df = get_influencial_features_df(journeys_count_df)
    features_df = pd.concat([counts_df, influential_features_df], axis = 1)
    return features_df

def get_historic_features_df(features_df, timesteps):
    x = []
    for timestep in range(1, timesteps+1):
        historic_features_df = features_df.shift(timestep)
        historic_features_df.columns = ['{}(t-{})'.format(col, timestep) for col in historic_features_df.columns]
        x.append(historic_features_df)

    x.reverse()
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

def get_outputs(features_df, timesteps):
    y = features_df.values[timesteps:, :station_count*2]
    print('y shape: {}'.format(y.shape))

    return y

def prepare_data(journeys_count_df, timesteps):
    preprocess_journeys_count_df(journeys_count_df)
    features_df = get_features_df(journeys_count_df)
    x = get_historic_features_df(features_df, timesteps)
    x = scale_x(x)
    x = reshape_inputs(x, timesteps)
    y = get_outputs(features_df, timesteps)

    # split into input and outputs
    test_len = sum(features_df.index >= TEST_PREDICT_START_TIME)
    train_X, train_y = x[:-test_len], y[:-test_len]
    test_X, test_y = x[-test_len:], y[-test_len:]

    print('train_X shape: {}'.format(train_X.shape))
    print('test_X shape: {}'.format(test_X.shape))
    print('train_y shape: {}'.format(train_y.shape))
    print('test_y shape: {}'.format(test_y.shape))

    return train_X, train_y, test_X, test_y, features_df

def train_model(model, epochs=200):
    earlystopping = EarlyStopping(patience=10, monitor='val_loss',
                                  verbose=2, restore_best_weights=True)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=32,
                    validation_split=0.3, shuffle=False, verbose=2,
                    callbacks=[earlystopping])
    return history

def plot_training_history(model_name, model, history):
    title = '{} Training History'.format(model_name)
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.plot([], [], ' ', label='Val RMSE = {:.3f}'.format(model.evaluate(test_X, test_y, verbose = 0) ** 0.5))
    plt.legend(fontsize = 20)
    plt.title(title, size = 25, pad=20)
    plt.xlabel('No. Epochs', fontsize = 20)
    plt.ylabel('MSE', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(os.path.join('images', title), dpi = 200)

def make_prediction(model, features_df):
    print('RMSE (float) = {:.3f}'.format(model.evaluate(test_X, test_y, verbose = 0) ** 0.5))

    # Round to integer and convert negative number to 0
    y_pred = model.predict(test_X, verbose = 0).round(0).astype(int).clip(0)
    predict_df = pd.DataFrame(data = y_pred, index = features_df.index[features_df.index >= PREDICT_START_TIME], columns = features_df.columns[:station_count*2])
    print('RMSE (int) = {:.3f}'.format(get_rmse(predict_df.values.flatten(), test_y.flatten())))

    return predict_df

def postprocess_prediction(model_name, predict_df):
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

    filepath = 'data/processed/london_journeys_predict_with_2h_interval_{}.csv'.format(model_name)
    predict_df.to_csv(filepath, index = False)
    print('Save prediction to {}'.format(filepath))

journeys_count_df = pd.read_csv('data/processed/london_journeys_count_with_2h_interval.csv', parse_dates=['Time'])
station_count = len(journeys_count_df['Station ID'].unique()) # 771
train_X, train_y, test_X, test_y, features_df = prepare_data(journeys_count_df, timesteps = 7*12)

# x shape: (4296, 84, 1544)
# y shape: (4296, 1542)
# train_X shape: (2832, 84, 1544)
# test_X shape: (1464, 84, 1544)
# train_y shape: (2832, 1542)
# test_y shape: (1464, 1542)


"""LSTM"""
def build_lstm(x):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(x.shape[1], x.shape[2]),
                   activation = 'relu'))
    model.add(Dense(100))
    model.add(Dense(station_count*2))
    model.compile(loss="mse", optimizer="adam")
    return model

"""GRU"""
def build_gru(x):
    model = Sequential()
    model.add(GRU(units=50, input_shape=(x.shape[1], x.shape[2]),
                   activation = 'relu'))
    model.add(Dense(100))
    model.add(Dense(station_count*2))
    model.compile(loss="mse", optimizer="adam")
    return model

"""Bidirectional LSTM"""
def build_bi_lstm(x):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, input_shape=(x.shape[1], x.shape[2]),
                   activation = 'relu')))
    model.add(Dense(100))
    model.add(Dense(station_count*2))
    model.compile(loss="mse", optimizer="adam")
    return model

scores_df = pd.DataFrame(columns = ['data', 'model', 'RMSE'])
get_models = [build_lstm, build_gru, build_bi_lstm]
model_names = ['LSTM', 'GRU', 'Bi-LSTM']
epochs = 200
for i in range(10):
    for get_model, name in zip(get_models, model_names):
        model = get_model(train_X)
        train_model(model, epochs)
        scores_df.loc[scores_df.shape[0]] = ['Train', name, model.evaluate(train_X, train_y, verbose = 0) ** 0.5]
        scores_df.loc[scores_df.shape[0]] = ['Test', name, model.evaluate(test_X, test_y, verbose = 0) ** 0.5]

scores_df.describe()
scores_df.to_csv('data/model_performence.csv', index = False)

# scores_df = pd.read_csv('data/model_performence.csv')
fig = sns.catplot(data = scores_df, kind = 'box', x = 'model', y = 'RMSE', hue = 'data', height = 6, aspect = 1.4, legend=False)
plt.title('Scores Distribution Across Models', size = 25, pad=20)
plt.xlabel('Models', fontsize = 20)
plt.ylabel('RMSE', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20, bbox_to_anchor=(1.3, 0.65))
fig.savefig('images/scores_boxplot', dpi = 200)

mean_scores_df = scores_df.groupby(['data', 'model'])['RMSE'].mean().reset_index()
mean_scores_df.loc[mean_scores_df.shape[0]] = ['Train', '7DMA', P7MA_train_RMSE]
mean_scores_df.loc[mean_scores_df.shape[0]] = ['Test', '7DMA', P7MA_test_RMSE]

mean_scores_df
mean_scores_df.to_csv('data/model_mean_RMSE.csv', index = False)
mean_scores_df

fig = sns.catplot(data = mean_scores_df, x = 'model', y = 'RMSE',kind = 'bar',
    order=['7DMA', 'LSTM', 'GRU', 'Bi-LSTM'],
    hue = 'data', hue_order=['Train', 'Test'], height = 6, aspect = 1.4, legend=False)
plt.title('Mean RMSEs Across Models', size = 25, pad=20)
plt.xlabel('Models', fontsize = 20)
plt.ylabel('Mean RMSE', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20, bbox_to_anchor=(1, 0.65))
for p in fig.ax.patches:
    fig.ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+0.05),
                    ha='center', va='bottom', color= 'black', fontsize=15)
fig.savefig('images/scores_mean', dpi = 200)



lstm = build_lstm(train_X)
history = train_model(lstm)
plot_training_history('LSTM', lstm, history)
predict_df = make_prediction(lstm, features_df)
postprocess_prediction('LSTM', predict_df)
#
# gru = build_gru(train_X)
# history = train_model(gru)
# plot_training_history('GRU', gru, history)
# predict_df = make_prediction(gru, features_df)
# postprocess_prediction('GRU', predict_df)
#
# bi_lstm = build_bi_lstm(train_X)
# history = train_model(bi_lstm)
# plot_training_history('Bidirectional LSTM', bi_lstm, history)
# predict_df = make_prediction(bi_lstm, features_df)
# postprocess_prediction('Bidirectional LSTM', predict_df)

# # Make prediction given a time
# journeys_predict_df = pd.read_csv('data/processed/london_journeys_predict_with_2h_interval_LSTM.csv', parse_dates=['Time'])
# time = PREDICT_START_TIME
# records = journeys_predict_df[(journeys_predict_df['Time'] == time)]
# records = {row['Station ID']: {'in': int(row['In']), 'out': int(row['Out'])} for index, row in records.iterrows()}
