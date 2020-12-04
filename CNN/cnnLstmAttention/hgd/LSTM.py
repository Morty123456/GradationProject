from keras.layers import Input, LSTM, Conv1D, Bidirectional
from keras.models import Sequential
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


path = '../pollution.csv'
rate = [0.7, 0.2, 0.1]
train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess.prepro(path, rate)
model = baselineModel()
model.fit(train_x, train_y, epochs=5, batch_size=64, validation_data=[valid_x, valid_y])
score = model.evaluate(test_x, test_y)
print(score[0])
print(score[1])
plot_model(model=model, to_file='attentionLSTM.png', show_shapes=True)