from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils, plot_model
import preprocess
import numpy as np

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=r'../data\0HP', length=2048, number=1000, normal=True, rate=[0.7, 0.2, 0.1], enc=True, enc_step=28)

# np.newaxis 增加一个维度，把二维转化为三维
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]

# 输入的是2维的
input_shape = x_train.shape[1:]

def baseline_model():
    model = Sequential()
    # model.add(Conv1D(32, 8, input_shape=(2048, 1)))
    model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    plot_model(model, to_file='./cnn_hgd.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(x=x_train, y=y_train, batch_size=128, epochs=50, validation_data=(x_valid, y_valid))
score = model.evaluate(x=x_test, y=y_test)
print(score[0])
print(score[1])