from keras.layers import Input, LSTM, Conv1D, Bidirectional
from keras.models import Sequential
from keras.utils import plot_model

from keras.layers.core import *
import preprocess


def baselineModel():
    model = Sequential()
    # 输入的维度 20*7
    inputs = Input(shape=(20, 7))
    # inputs = train_X.shape
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model


path = '../pollution.csv'
train_X, train_Y = preprocess.prepro(path)
model = baselineModel()
model.fit([train_X], train_Y, epochs=5, batch_size=64, validation_split=0.1)
plot_model(model=model, to_file='attentionLSTM.png', show_shapes=True)