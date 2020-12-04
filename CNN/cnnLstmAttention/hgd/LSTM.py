from keras.layers import Input, LSTM, Conv1D, Bidirectional
from keras.models import Sequential, Model
from keras.utils import plot_model

from keras.layers.core import *
import preprocess


def baselineModel():
    model = Sequential()
    # 输入的维度 20*7
    # inputs = Input(shape=(20, 7))
    # inputs = train_X.shape
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model
def baselineModel2():
    inputs = Input(shape=(20, 7))
    # 输出为 20*64
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same',必要时使用0进行填充
    # dropout: 防止过拟合,神经元以0.3的概率停止工作
    x = Dropout(0.3)(x)
    # 输出依旧为 20*64
    # Bidirectional RNN 双向循环网络
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_mul = Flatten()(lstm_out)
    output = Dense(1, activation='sigmoid')(lstm_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model



path = '../pollution.csv'
rate = [0.7, 0.2, 0.1]
train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess.prepro(path, rate)
model = baselineModel2()
model.compile(optimizer='adam', loss='mse')
model.fit([train_x], train_y, epochs=5, batch_size=64, validation_data=[valid_x, valid_y])
# model.fit(train_x, train_y, epochs=5, batch_size=64, validation_split=0.1)
predict = model.predict(test_x)
loss = 0
for i in range(len(predict)):
    loss = loss + (predict[i]-test_y[i])*(predict[i]-test_y[i])
print(loss)
# print(len(test_y), len(predict))
plot_model(model=model, to_file='attentionLSTM.png', show_shapes=True)