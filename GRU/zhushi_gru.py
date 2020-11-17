import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

CSV_PATH = './1.csv'
maotai = pd.read_csv(CSV_PATH)

# 得到的数据都 list 格式
training_set = maotai.iloc[0:2426-300, 2:3].values
test_set = maotai.iloc[2426-300:, 2:3].values
# print(len(training_set), len(test_set))
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# print(len(x_train), len(x_train[0]))
# print(y_train)
np.random.seed(7)
# 乱序排列
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# print(len(x_train))
x_train, y_train = np.array(x_train), np.array(y_train)
# print(len(x_train[0]))
# print(x_train)
# 把二维数组切分成三维数组
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
# print(len(x_train), len(x_train[0]), len(x_train[0][0]))
for i in range(60, len(test_set)):
    x_test.append(test_set[i-60:i, 0])
    y_test.append(test_set[i, 0])
# 测试数据
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    GRU(80, return_sequences=True),
    # 防止过拟合，让某个上神经元以 0.2 的概率停止工作
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    # 全连接神经网络
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')

checkpoint_save_path = "./checkpoint/zhushi/stock.ckp"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=True,
                                                save_best_pnly=True,
                                                monitor='val_loss')
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])
model.summary()

# loss = history.history['loss']
# val_loss = history.history['val_loss']
predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(test_set[60:])

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predict MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()

