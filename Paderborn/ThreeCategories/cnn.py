import Paderborn.ThreeCategories.preprocess as preprocess
import Paderborn.get_classification_accuracy as getClassification
import numpy as np
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation
from keras.models import Sequential
from keras.regularizers import l2

# 训练参数
batch_size = 128
epochs = 100
num_classes = 3
length = 1800
number = 200
rate = [0.7, 0.2, 0.1]
path = r'D:\Data\Paderborn\data'

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path, length=length, number=number, rate=rate
)
# np.newaxis 增加一个维度，把二维数据转为三维
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
# 输入的数据是2维的
input_shape = x_train.shape[1:]

def cnn():
    model =Sequential()
    model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(4))
    # model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4),
    #                  input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(4))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = cnn()
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
y_predict = model.predict(x_test)
score, res = getClassification.classification(y_predict, y_test)
print(score)