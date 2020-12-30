import Paderborn.NineCtaegories.preprocess as preprocess
import Paderborn.get_classification_accuracy as getClassification
import numpy as np
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import plot_model


# 训练参数
batch_size = 64
epochs = 1
num_classes = 9
length = 1800
number = 1000
rate = [0.7, 0.2, 0.1]
path = r'D:\Data\Paderborn\data'

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.preprocess(
    d_path=path, length=length, number=number, rate=rate
)
print(len(x_train), len(x_train[0]))
print(x_train[0:2])
# np.newaxis 增加一个维度，把二维数据转为三维
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
# 输入的数据是2维的
input_shape = x_train.shape[1:]

def cnn():
    model =Sequential()
    model.add(Conv1D(filters=9, kernel_size=3, strides=8, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(4))
    # model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4),
    #                  input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(4))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    plot_model(model, to_file='./cnn_nine.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = cnn()
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
y_predict = model.predict(x_test)
score, socreForOne, predict_count, test_count, correctClassification = getClassification.classification_detailded(y_predict, y_test)
print(score)
print(socreForOne)
print(predict_count)
print(test_count)
print(correctClassification)