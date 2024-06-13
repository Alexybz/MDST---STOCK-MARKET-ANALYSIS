import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime
# import plotly.graph_objects as go
import pandas_ta as ta

data = yf.download(tickers = 'NVDA')

plt.figure(figsize=(16,8))
plt.plot(data["Adj Close"], color = 'black', label = 'Price')

Aroon = ta.aroon(data.High, data.Low)
data = pd.concat([data, Aroon], axis = 1)

data['RSI']=ta.rsi(data.Close, length=15)
data['EMAF']=ta.ema(data.Close, length=20)
data['EMAM']=ta.ema(data.Close, length=100)
data['EMAS']=ta.ema(data.Close, length=150)
data['ATR']=ta.atr(data['High'], data['Low'], data['Close'], length=14)
data['MFI']=ta.mfi(data.High, data.Low, data.Close, data.Volume)


data['Target'] = data['Adj Close']-data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace = True)
plt.figure(figsize=(16,8))
# plt.plot(data["Adj Close"], color = 'black', label = 'Price')
# plt.plot(data["RSI"], color = 'black', label = 'Price')
plt.plot(data["EMAF"], color = 'black', label = 'Price')
plt.plot(data["EMAM"], color = 'green', label = 'Price')
plt.plot(data["EMAS"], color = 'blue', label = 'Price')

plt.plot(data["Adj Close"], color = 'black', label = 'Price')
data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
data_set = data

column = len(data.columns) - 3

target_col = "TargetNextClose"

from sklearn.preprocessing import MinMaxScaler
sc_data = MinMaxScaler(feature_range=(0,1))
sc_response = MinMaxScaler(feature_range=(0,1))

response_scaled = sc_response.fit_transform(data_set[[target_col]])
# print(response_scaled)
# print(data_set.loc[:, data_set.columns != target_col])
# print(response_scaled)
data_set_scaled = sc_data.fit_transform(data_set.loc[:, data_set.columns != target_col])
print(data_set_scaled.shape)
# print(data_set_scaled)
print(len(data_set_scaled))
data_set_scaled_new = []
for i in range(len(data_set_scaled)):
    data_set_scaled_new.append([])
    # print(response_scaled[i][0])

    # print(data_set_scaled[i])
    data_set_scaled_new[i] = np.append(data_set_scaled[i],response_scaled[i])
    # print(data_set_scaled_new[i])
    # print("#"*80)

data_set_scaled_new = np.asarray(data_set_scaled_new)
print(data_set_scaled_new)

X = []
backcandles = 30
print(data_set_scaled_new.shape[0])
for j in range (column):
  X.append([])
  for i in range(backcandles, data_set_scaled_new.shape[0]):
    X[j].append(data_set_scaled_new[i-backcandles:i,j ])

# print(X)
X = np.moveaxis(X, [0], [2])
# print(X)


X, yi = np.array(X), np.array(data_set_scaled_new[backcandles:, -1])
print(yi.shape)
y = np.reshape(yi, (len(yi),1))

print(X)
print(X.shape)
print(y)
print(y.shape)

#Splitting data into training and testing sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
#tf.random.set_seed(20)
np.random.seed(10)

lstm_input = Input(shape=(backcandles, column), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=20, shuffle=False, validation_split = 0.1)

y_pred = model.predict(X_test)
#y_pred=np.where(y_pred > 0.43, 1,0)
for i in range(10):
    print(y_pred[i], y_test[i])

print(sc_response.inverse_transform(y_test)[0])
print(sc_response.inverse_transform(y_pred)[0])

tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_test))))

plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'green', label = 'pred')
plt.legend()
plt.show()
